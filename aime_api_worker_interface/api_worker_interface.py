# Copyright (c) AIME GmbH and affiliates. Find more info at https://www.aime.info/api
#
# This software may be used and distributed according to the terms of the MIT LICENSE

import time
from datetime import datetime, timedelta
import sys
import signal
import tty
import termios
import gc
import select
import threading
import requests

from multiprocessing import Barrier, Event, Lock
from multiprocessing.managers import SyncManager
from concurrent.futures import ThreadPoolExecutor


import socket
import io

import base64
from PIL.PngImagePlugin import PngInfo
import json
import pkg_resources


SYNC_MANAGER_BASE_PORT  =  10042
SYNC_MANAGER_AUTH_KEY   = b"aime_api_worker"
SERVER_PARAMETERS = ['job_id', 'start_time', 'start_time_compute']
DEFAULT_IMAGE_METADATA = [
    'prompt', 'negative_prompt', 'seed', 'base_steps', 'refine_steps', 'scale', 
    'aesthetic_score', 'negative_aesthetic_score', 'img2img_strength', 'base_sampler', 
    'refine_sampler', 'base_discretization', 'refine_discretization'
                        ]


class WorkerSyncManager(SyncManager):
    pass


class APIWorkerInterface():
    """Interface for deep learning models to communicate with AIME API Server.

    Args:
        api_server (str): Address of API Server. Example: 'http://api.aime.team'.
        job_type (str): Type of job . Example: "stable_diffusion_xl_txt2img".
        auth_key (str): key to authorize worker to connect with API Server.
        gpu_id (int, optional): ID of GPU the worker runs on. Defaults to 0.
        world_size (int, optional): Number of used GPUs the worker runs on. Defaults to 1.
        rank (int, optional): ID of current GPU if world_size > 1. Defaults to 0.
        gpu_name (str, optional): Name of GPU the worker runs on. Defaults to None.
        progress_received_callback (callable, optional): Callback function with http response as argument, 
            called when API server sent response to send_progress(..). Defaults to None.
        progress_error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError as argument, 
            called when API server didn't send response from send_progress(..). Defaults to None.
        image_metadata_params (list, optional): Parameters specific for the used image generator model to add as metadata to the generated image. 
            Fixed parameters are Artist, ProcessingSoftware, Software, ImageEditingSoftware = AIME API <endpoint_name>. 
            Defaults to aime_api_worker_interface.DEFAULT_IMAGE_METADATA. 

    Attributes:
        progress_data_received (bool): True, if API server sent response to send_progress(), 
            False while progress data is being transmitted or if an error occured.

    Examples:
        Minimal example, instantiate the api_worker with URL to the API server, 
        job type and auth_key. Waiting for and receiving job data and sending job result:

        .. highlight:: python
        .. code-block:: python            

            from aime_api_worker_interface import APIWorkerInterface
                
            api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
            while True:
                job_data = api_worker.job_request()
                output = do_deep_learning_worker_calculations(job_data, ...)
                api_worker.send_job_results(output)

        Example usage with progress:

        .. highlight:: python
        .. code-block:: python

            from aime_api_worker_interface import APIWorkerInterface           

            api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
            while True:
                job_data = api_worker.job_request()
                
                for step in deep_learning_worker_calculation:
                    progress_in_percent = round(step*100/len(deep_learning_worker_calculation))
                    progress_data = do_deep_learning_worker_calculation_step(job_data, ...)
                    if api_worker.progress_data_received:
                        api_worker.send_progress(progress_in_percent, progress_data)
                output = get_result()
                api_worker.send_job_results(output)

        Example usage with callback:

        .. highlight:: python
        .. code-block:: python

            from aime_api_worker_interface import APIWorkerInterface

            def progress_callback(api_worker, progress, progress_data):
                if api_worker.progress_data_received:
                    api_worker.send_progress(progress, progress_data)


            api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
            callback = Callback(api_worker)
            
            while True:
                job_data = api_worker.job_request()
                output = do_deep_learning_worker_calculation(job_data, progress_callback, api_worker, ...)
                api_worker.send_progress(progress, progress_data)

        Example usage with callback class:

        .. highlight:: python
        .. code-block:: python

            from aime_api_worker_interface import APIWorkerInterface

            class Callback():

                def __init__(self, api_worker):
                    self.api_worker = api_worker


                def progress_callback(self, progress, progress_data):
                    if self.api_worker.progress_data_received:
                        self.api_worker.send_progress(progress, progress_data)
                
                def result_callback(self, result):
                    self.api_worker.send_job_results(result) 


            api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
            callback = Callback(api_worker)
            
            while True:
                job_data = api_worker.job_request()
                do_deep_learning_worker_calculation(job_data, callback.result_callback, callback.progress_callback, ...)
        
    """
    manager = None
    barrier = None
    error_event = Event()

    def __init__(
        self, 
        api_server, 
        job_type, 
        auth_key, 
        gpu_id=0, 
        world_size=1, 
        rank=0, 
        gpu_name=None, 
        image_metadata_params=DEFAULT_IMAGE_METADATA,
        print_server_status = True,
        request_timeout = 60,
        worker_version=0,
        exit_callback=None
        ):
        """Constructor

        Args:
            api_server (str): Address of API Server. Example: 'http://api.aime.team'.
            job_type (str): Type of job . Example: "stable_diffusion_xl_txt2img".
            auth_key (str): key to authorize worker to connect with API Server.
            gpu_id (int, optional): ID of GPU the worker runs on. Defaults to 0.
            world_size (int, optional): Number of used GPUs the worker runs on. Defaults to 1.
            rank (int, optional): ID of current GPU if world_size > 1. Defaults to 0.
            gpu_name (str, optional): Name of GPU the worker runs on. Defaults to None.
            image_metadata_params (list, optional): Parameters to add as metadata to images (Currently only 'PNG'). Defaults to [
                'prompt', 'negative_prompt', 'seed', ...]
            print_server_status (bool, optional): Whether the server status is printed at start. Defaults to True
        """        
        self.api_server = api_server
        self.job_type = job_type
        self.auth_key = auth_key
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.rank = rank
        self.gpu_name = gpu_name
        self.image_metadata_params = image_metadata_params
        self.print_server_status = print_server_status
        self.request_timeout = request_timeout
        self.worker_version = worker_version
        self.exit_callback = exit_callback


        self.progress_input_params = dict()
        self.awaiting_job_request = False
        self.__custom_callback = None
        self.__custom_error_callback = None
        self.progress_data_received = True
        self.print_idle_string_thread = None
        self.pool_executor = ThreadPoolExecutor(max_workers=None)
        self.lock = Lock()
        self.exit_event = Event()
        self.new_job_event = Event()
        self.__init_manager_and_barrier()
        self.__current_job_cmds = APIWorkerInterface.manager.dict() # key job_id
        self.old_terminal_settings = termios.tcgetattr(sys.stdin)

        self.version = self.get_version()
        self.worker_name = self.__make_worker_name()

        self.register_interrupt_signal_handler()
        self.keyboard_input_listener_thread = self.init_and_start_keyboard_input_listener_thread()
        
        self.async_check_server_connection(terminal_output = print_server_status)

            
    def job_request(self):
        """Worker requests a single job from the API Server on endpoint route /worker_job_request. 

        Does call job_batch_request() with max_job_batch size 1 and returns the first job.

        See job_batch_request() for more information.

        Returns:
           dict: job data with worker [INPUT] parameters received from API server.
        """
        self.job_batch_request(1)
        return self.get_current_job_data()


    def job_batch_request(self, max_job_batch, wait_for_response=True, callback=None, error_callback=None):
        """Worker requests a job batch from API Server on endpoint route /worker_job_request.

        If there is no client job offer within the job_timeout = request_timeout * 0.9 the API server 
        responds with 'cmd':'no_job' and the worker requests a job again on endpoint route/worker_job_request.
         
        In MultGPU-Mode (world_size > 1) only rank 0 will get the job_data.

        Args:
            max_job_batch (int): Maximum job batch size to request. 
            wait_for_response (bool, optional): Whether the method blocks until the API Server response is received. If set to False, callback and error_callback are utilized to get the response. Default to True.
            callback (callable, optional): Callback function with API server response as argument. 
                Called, when job request is received by the API Server, if wait_for_response is set to True. Defaults to None.
            error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError or 
                http response with :status_code == 503: as argument. Called when API server replied with error, if wait_for_response is set to True. Defaults to None.

        Returns:
            list: List of job data with worker [INPUT] parameters received from API server.

        Examples:

            Example job data:

            .. highlight:: python
            .. code-block:: python

                response_data = {
                    'wait_for_result': False,
                    'endpoint_name': 'stable_diffusion_xl_txt2img',
                    'start_time': 1700430052.505548,
                    'start_time_compute': 1700430052.5124364},
                    'cmd': 'job',
                    'job_data': [{
                        'job_id': 'JID1',
                        'prompt': 'prompt',
                        ... 
                    }],
                    'progress_descriptions': {
                        'progress_images': {
                            'type': 'image_list', 'image_format': 'JPEG', 'color_space': 'RGB'
                        }
                    }, 
                    'output_descriptions': {
                        'images': {'type': 'image_list', 'image_format': 'JPEG', 'color_space': 'RGB'},
                        'seed': {'type': 'integer'},
                        'prompt': {'type': 'string'},
                        'error': {'type': 'string'}
                    }
                }
        """
        if wait_for_response:
            job_cmd = dict()
            if self.rank == 0:
                have_job = False
                counter = 0
                while not have_job:
                    request = {
                        'auth': self.worker_name,
                        'job_type': self.job_type,
                        'auth_key': self.auth_key,
                        'version': self.version,
                        'worker_version': self.worker_version,
                        'request_timeout': self.request_timeout,
                        'max_job_batch': max_job_batch,
                    }
                    try:
                        response = self.__fetch('/worker_job_request', request)

                        if response.status_code == 503:
                            self.check_periodically_if_server_online()
                            continue
                    except requests.exceptions.ConnectionError:
                        self.check_periodically_if_server_online()
                        continue
                    response_output_str = '! API server responded with {cmd}: {msg}'
                    if response.status_code == 200:
                        job_cmd = response.json()
                        job_cmd['last_activity'] = time.time()
                        cmd = job_cmd.get('cmd')
                        msg = job_cmd.get('msg', 'unknown')
                        if cmd == 'job':
                            have_job = True
                        elif cmd == 'no_job':
                            continue
                        elif cmd == 'error':
                            print(response_output_str.format(cmd=cmd, msg=msg))
                            self.error_event.set()
                            if error_callback:
                                self.__async_job_batch_request_error_callback_wrapper(callback, response)
                            break
                        elif cmd == 'warning':
                            counter += 1
                            self.check_periodically_if_server_online()
                            if counter > 3:
                                print(response_output_str.format(cmd=cmd, msg=msg))
                                self.error_event.set()
                                break
                        else:
                            print(response_output_str.format(cmd='unknown command', msg=cmd))
                            self.error_event.set()
                            break
            if self.world_size > 1:
                # hold all GPU processes here until we have a new job
                APIWorkerInterface.barrier.wait()
            if self.error_event.is_set():
                print('Error Event')
                self.gracefully_exit()
            else:
                job_batch_data = job_cmd.get('job_data', [])                
                for job_data in job_batch_data:
                    if job_data:
                        job_cmd['job_data'] = job_data
                        self.__current_job_cmds[job_data.get('job_id')] = job_cmd
                if callback:
                    self.__async_job_batch_request_callback_wrapper(callback, job_batch_data)
                return job_batch_data
        else:
            self.awaiting_job_request = True
            self.pool_executor.submit(self.job_batch_request, max_job_batch, True, callback, error_callback)


    def job_request_generator(self, max_job_batch):
        """Generator yielding the related job_batch_data whenever there are new job_requests or an empty list if there are running jobs but no new job requests.
        Blocking, if there are no running jobs and no new job requests to spare hardware resources.

        Args:
            max_job_batch (int): Maximum of parallel running jobs

        Yields:
            list: job_batch_data of new job requests or empty list
        """

        while True:
            with self.lock:
                remaining_job_batch = max_job_batch - len(self.__current_job_cmds)
                if remaining_job_batch > 0 and not self.awaiting_job_request:
                    self.job_batch_request(remaining_job_batch, wait_for_response=False, callback=self.__generator_callback)

                jobs_to_start = [job_cmd.get('job_data') for job_cmd in self.__current_job_cmds.values() if job_cmd.get('awaiting_yield')]
                for job_data in jobs_to_start:
                    job_cmd = self.__current_job_cmds[job_data.get('job_id')]
                    job_cmd['awaiting_yield'] = False
                    self.__current_job_cmds[job_data.get('job_id')] = job_cmd # SyncManager().dict() doesn't support direct assignment of nested dictionary value
            yield jobs_to_start

                
            if self.have_all_jobs_finished():
                self.wait_for_job()


    def send_job_results(self, results, job_data={}, job_id=None, wait_for_response=True, callback=None, error_callback=None):
        """Process/convert job results and send it to API Server on route /worker_job_result.

        Args:
            results (dict): worker [OUTPUT] result parameters (f.i. 'image', 'images' or 'text').
                Example results: ``{'images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}``
            job_data (dict, optional): To use different job_data than the received one or to identify the related job if no job_id is given. Defaults to {}.
            job_id (str, optional): To identify the related job. Defaults to None.
            wait_for_response (bool, optional): Whether the methods blocks until the API Server response is received. Default to True.
            callback (callable, optional): Callback function with API server response as argument. 
                Called, when job result is received by the API Server, if wait_for_response is set to True. Defaults to None.
            error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError or 
                http response with :status_code == 503: as argument. Called when API server replied with error, if wait_for_response is set to True. Defaults to None.

        Returns:
            requests.models.Response: Http response from API server to the worker.

        Examples:
            Example response.json(): 
            
            .. code-block:: 

                API Server received data without problems:          {'cmd': 'ok'} 
                An error occured in API server:                     {'cmd': 'error', 'msg': <error message>} 
                API Server received data received with a warning:   {'cmd': 'warning', 'msg': <warning message>}
        """
        if wait_for_response:
            if self.rank == 0:
                job_id = job_id or job_data.get('job_id')
                job_data = job_data or self.get_current_job_data(job_id)   
                results = self.__prepare_output(results, job_data, True)

                with self.lock:
                    if job_id:
                        self.__current_job_cmds.pop(job_id, None)
                    else:
                        self.__current_job_cmds.clear()
                results['auth_key'] = self.auth_key
                results['job_type'] = self.job_type
                while True:
                    try:
                        response =  self.__fetch('/worker_job_result', results)
                    except requests.exceptions.ConnectionError:
                        self.check_periodically_if_server_online()
                        continue
                    except Exception as exception:
                        if error_callback:
                            error_callback(exception)
                        else:
                            raise exception

                    if response.status_code == 200:
                        if callback:
                            callback(response)
                        return response
                    else:
                        if error_callback:
                            error_callback(exception)
                        self.check_periodically_if_server_online()
                        continue
        else:
            self.pool_executor.submit(self.send_job_results, results, job_data, job_id, True, callback, error_callback)


    def __job_result_callback(self, future, callback, error_callback):
        try:
            result = future.result()
            if callback:
                callback(result)
            return result
        except Exception as error:
            if error_callback:
                error_callback(error)


    def send_progress(
        self,
        progress,
        progress_data=None,
        progress_received_callback=None,
        progress_error_callback=None,
        job_data={},
        job_id=None
        ):
        """Processes/converts job progress information and data and sends it to API Server on route /worker_job_progress asynchronously 
        to main thread using Pool().apply_async() from multiprocessing.dummy. When Api server received progress data, 
        self.progress_data_received is set to True. Use progress_received_callback and progress_error_callback for response.

        Args:
            progress (int): current progress (f.i. percent or number of generated tokens)
            progress_data (dict, optional): dictionary with progress_images or text while worker is computing. 
                Example progress data: :{'progress_images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}:. Defaults to None.
            progress_received_callback (callable, optional): Callback function with API server response as argument. 
                Called when progress_data is received. Defaults to None.
            progress_error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError or 
                http response with :status_code == 503: as argument. Called when API server replied with error. Defaults to None.
            job_data (dict, optional): To use different job_data than the received one or to identify the related job if no job_id is given. Defaults to {}.
            job_id (str, optional): To identify the related job. Defaults to None.
            
        """
        job_id = job_id or job_data.get('job_id') 
        job_data = job_data or self.get_current_job_data(job_id)
        if self.rank == 0 and self.progress_data_received and not self.__get_current_job_cmd(job_id).get('wait_for_result', False):           
            payload = {parameter: job_data[parameter] for parameter in SERVER_PARAMETERS}
            payload.update(
                {
                    'job_type': self.job_type,
                    'auth_key': self.auth_key,
                    'progress': progress,
                    'progress_data': self.__prepare_output(progress_data, job_data, False)
                }
            )
            self.progress_data_received = False
            _ = self.__fetch_async('/worker_job_progress', payload, progress_received_callback, progress_error_callback)
            self.update_job_activity(job_id)


    def send_batch_progress(
        self,
        batch_progress,
        progress_batch_data,
        progress_received_callback=None,
        progress_error_callback=None,
        job_batch_data=[],
        job_batch_ids=None
        ):
        """Processes/converts job progress information and data and sends it to API Server on route /worker_job_progress asynchronously 
        to main thread using Pool().apply_async() from multiprocessing.dummy. When Api server received progress data, 
        self.progress_data_received is set to True. Use progress_received_callback and progress_error_callback for response.

        Args:
            batch_progress (list(int, int, ...)): current progress (f.i. percent or number of generated tokens)
            progress_batch_data (list(dict, dict, ...), optional): dictionary with progress_images or text while worker is computing. 
                Example progress data: :{'progress_images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}:. Defaults to None.
            progress_received_callback (callable, optional): Callback function with API server response as argument. 
                Called when progress_data is received. Defaults to None.
            progress_error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError or 
                http response with :status_code == 503: as argument. Called when API server replied with error. Defaults to None.
            job_batch_data (list(dict, dict, ...), optional): List of job datas to use different job_datas than the received one or to identify the related jobs, if no job_batch_ids are given. Default to [].
            job_batch_ids (list(int, int, ...), optional): List of job ids to identify the related jobs. Defaults to None.           
        """
        job_batch_ids = job_batch_ids or [job_data.get('job_id') for job_data in job_batch_data]
        job_batch_data = job_batch_data or [self.get_current_job_data(job_id) for job_id in job_batch_ids]
        if self.rank == 0 and self.progress_data_received:
            batch_payload = []
            for progress, progress_data, job_data in zip(batch_progress, progress_batch_data, job_batch_data):
                if not self.__current_job_cmds.get(job_data.get('job_id')).get('wait_for_result'):
                    payload = {parameter: job_data.get(parameter) for parameter in SERVER_PARAMETERS}
                    payload.update(
                        {
                            'job_type': self.job_type,
                            'auth_key': self.auth_key,
                            'progress': progress, 
                            'progress_data': self.__prepare_output(progress_data, job_data, False)
                        }
                    )
                    batch_payload.append(payload)
            self.progress_data_received = False
            _ = self.__fetch_async('/worker_job_progress', batch_payload, progress_received_callback, progress_error_callback)
            for job_id in job_batch_ids:
                self.update_job_activity(job_id)


    def init_and_start_keyboard_input_listener_thread(self):
        keyboard_input_listener_thread = threading.Thread(target=self.keyboard_input_listener)
        keyboard_input_listener_thread.start()
        return keyboard_input_listener_thread


    def register_interrupt_signal_handler(self):
        signal.signal(signal.SIGUSR1, self.signal_handler)


    def signal_handler(self, sig, frame):
        self.gracefully_exit()

        
    def keyboard_input_listener(self, refresh_interval=1, send_signal=False):
        try:
            APIWorkerInterface.barrier.wait()
            tty.setcbreak(sys.stdin.fileno())
            while True:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    keyboard_input = sys.stdin.read(1)
                    if keyboard_input in ('q', '\x1b'): # x1b is ESC
                        self.exit_event.set()
                        self.new_job_event.set()
                        break
                time.sleep(refresh_interval)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            signal.raise_signal(signal.SIGUSR1)
            

    def get_current_job_data(self, job_id=None):
        """Get the job_data of current job to be processed

        Args:
            job_id (string, optional): For single job processing (max_job_batch=1) the job_id is not required.
                             For batch job processing (max_job_batch>1) the job_id is required to specify 
                             the job and the job_data is going to be returned. Defaults to None.

        Returns:
            dict: the job_data of the current only job or of the job with the given job_id
        """
        return self.__get_current_job_cmd(job_id).get('job_data')


    def __get_current_job_cmd(self, job_id=None):
        if job_id:
            job_cmd = self.__current_job_cmds.get(job_id)
            if job_cmd is None:
                raise ValueError(f'No current job with job_id: {job_id}')
            return job_cmd

        elif len(self.__current_job_cmds) == 1:
            return self.__current_job_cmds.values()[0]
        elif len(self.__current_job_cmds) > 1:
            raise Exception("More then one running jobs: job_id argument required!")
        else:
            return None


    def get_current_job_batch_data(self):
        """get the job_datas of the current batch as list 

        Returns:
            list: the list of job_datas of the current jobs to be processed
        """
        return [job_cmd.get('job_data') for job_cmd in self.__current_job_cmds.values() if not job_cmd.get('awaiting_yield')]


    def get_job_batch_parameter(self, param_name):
        """get the job_data parameter values for a specific parameter as value array

        Returns:
            list: The value list of the parameter across the batch
        """       
        return [job_data.get(param_name) for job_data in self.get_current_job_batch_data()]      


    def has_job_finished(self, job_data={}, job_id=None):
        """Check if specific job has been addressed with a send_job_results and is thereby finished

        Args:
            job_data: job_data of the job to check

        Returns:
            bool: True if job has send job_results, False otherwise
        """
        return not bool(self.get_current_job_data(job_id or job_data.get('job_id')))


    def check_for_unresponsive_jobs(self, timeout=60):
        unresponsive_jobs = []
        if self.__current_job_cmds:
            for job_id, job_cmd in self.__current_job_cmds.items():
                if (not job_cmd.get('wait_for_result') and not job_cmd.get('awaiting_yield')) and ((time.time() - job_cmd.get('last_activity', 0)) > timeout):
                    unresponsive_jobs.append(job_id)
        return unresponsive_jobs
        #return [job_id for job_id, job_cmd in self.__current_job_cmds.items() if (not job_cmd.get('wait_for_result') and not job_cmd.get('awaiting_yield')) and ((time.time() - job_cmd.get('last_activity', 0)) > timeout)]


    def update_job_activity(self, job_id):
        job_cmd = self.__current_job_cmds.get(job_id)
        if job_cmd:
            job_cmd['last_activity'] = time.time()
            with self.lock:
                self.__current_job_cmds[job_id] = job_cmd


    def have_all_jobs_finished(self):
        """check if all jobs have been addressed with a send_job_results and are therefore finished

        Returns:
            bool: True if all jobs have send job_results, False otherwise
        """                  
        return not bool(self.__current_job_cmds)


    def get_canceled_job_ids(self):
        return [job_id for job_id, job_cmd in self.__current_job_cmds.items() if job_cmd.get('canceled')]


    def is_job_canceled(self, job_id=None):
        return self.__get_current_job_cmd(job_id).get('canceled', False)
    

    def wait_for_job(self):
        """Wait until self.new_job_event.set() is called while periodically printing idle string in a non-blocking background thread.
        """
        self.print_idle_string_thread = threading.Thread(target=self.print_idle_string)
        self.print_idle_string_thread.start()
        self.new_job_event.wait()
        self.new_job_event.clear()


    def gracefully_exit(self):
        """Gracefully exit the application while cleaning resources and threads with the possibility to call the custom callback exit_callback given in init of APIWorkerInterface().
        """
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
        if self.keyboard_input_listener_thread.is_alive():
            self.keyboard_input_listener_thread.join()
        if self.print_idle_string_thread:
            self.print_idle_string_thread.join()
        if APIWorkerInterface.barrier:
            APIWorkerInterface.barrier.wait()
        del self.__current_job_cmds
        APIWorkerInterface.manager.shutdown()
        if self.exit_callback:
            self.exit_callback()
        gc.collect()
        exit('\nGood bye!')


    def __async_job_batch_request_callback_wrapper(self, callback, job_batch_data):
        self.awaiting_job_request = False
        if callback:
            callback(job_batch_data)
        self.new_job_event.set()
        

    def __async_job_batch_request_error_callback_wrapper(self, callback, response):
        if callback:
            callback(response)
        else:
            raise requests.exceptions.ConnectionError(response)
        if self.error_event.is_set():
            exit()


    def __generator_callback(self, job_batch_data):
        
        for job_data in job_batch_data:
            job_cmd = self.__current_job_cmds[job_data.get('job_id')]
            job_cmd['awaiting_yield'] = True
            self.__current_job_cmds[job_data.get('job_id')] = job_cmd # SyncManager().dict() doesn't support direct assignment of nested dictionary value


    def pop_progress_input_params(self, job_id=None):
        return self.__get_current_job_cmd(job_id).pop('progress_input_params', None)


    def get_progress_input_params_batch(self):
        return [job_cmd.pop('progress_input_params') for job_cmd in self.__current_job_cmds.values() if job_cmd.get('progress_input_params')]


    def print_idle_string(self, refresh_interval=1):
        dot_string = self.dot_string_generator()
        print() # get cursor to next line
        while not self.__current_job_cmds:
            print(f'\033[F\033[KWorker idling{next(dot_string)}')
            time.sleep(refresh_interval)
            if self.exit_event.is_set():
                break


    def async_check_server_connection(
        self,
        check_server_callback=None,
        check_server_error_callback=None,
        terminal_output=True
        ):
        """Non blocking check of Api server status on route /worker_check_server_status using Pool().apply_async() from multiprocessing.dummy 

        Args:
            check_server_callback (callable, optional): Callback function with with API server response as argument. 
                Called after a successful server check. Defaults to None.
            check_server_error_callback (callable, optional): Callback function with requests.exceptions.ConnectionError as argument
                Called when server replied with error. Defaults to None.
            terminal_output (bool, optional): Prints server status to terminal if True. Defaults to True.
        """
        self.__custom_callback = check_server_callback if check_server_callback else None
        self.__custom_error_callback = check_server_error_callback if check_server_error_callback else None

        self.print_server_status = terminal_output
        payload =  {'auth_key': self.auth_key, 'job_type': self.job_type}
        self.__fetch_async('/worker_check_server_status', payload)


    def check_periodically_if_server_online(self, interval_seconds=1):
        """Checking periodically every interval_seconds=1 if server is online by post requests on route /worker_check_server_status.

        Args:
            interval_seconds (int): Interval in seconds to update check if server is online

        Returns:
            bool: True if server is available again
        """
        if self.rank == 0:
            server_offline = False
            start_time = datetime.now()
            dot_string = self.dot_string_generator()
            while True:
                try:
                    response = self.__fetch('/worker_check_server_status', {'auth_key': self.auth_key, 'job_type': self.job_type})
                    if response.status_code == 200:
                        if server_offline:
                            print('\nServer back online')
                        return True
                    else:
                        self.__print_server_offline_string(start_time, dot_string)
                        server_offline = True
                        time.sleep(interval_seconds)
                except requests.exceptions.ConnectionError:
                    self.__print_server_offline_string(start_time, dot_string)
                    server_offline = True
                    time.sleep(interval_seconds)


    def get_pnginfo_metadata(self, job_data):
        """Parses and returns image metadata from job_data.

        Returns:
            PIL.PngImagePlugin.PngInfo: PngInfo Object with metadata for PNG images
        """        
        metadata = PngInfo()
        for parameter_name in self.image_metadata_params:
            parameter = job_data.get(parameter_name)
            if parameter:
                metadata.add_text(parameter_name, str(parameter))
        aime_str = f'AIME API {self.__current_job_cmds.get(job_data.get("job_id")).get("endpoint_name", self.job_type)}'
        metadata.add_text('Artist', aime_str)
        metadata.add_text('ProcessingSoftware', aime_str)
        metadata.add_text('Software', aime_str)
        metadata.add_text('ImageEditingSoftware', aime_str)
        
        return metadata

    def get_binary(self, job_data, attrib):
        base64_data = job_data[attrib].split(',')[1]
        return base64.b64decode(base64_data)
    
    def get_binary_format(self, job_data, attrib):
        return job_data[attrib].split(',')[0]

    def get_exif_metadata(self, image, job_data):
        metadata = {str(parameter_name): job_data.get(parameter_name) for parameter_name in DEFAULT_IMAGE_METADATA}
        exif = image.getexif()
        exif[0x9286] = json.dumps(metadata) # Comment Tag
        aime_str = f'AIME API {self.__current_job_cmds.get(job_data.get("job_id")).get("endpoint_name", self.job_type)}'
        exif[0x013b] = aime_str # Artist
        exif[0x000b] = aime_str # ProcessingSoftware
        exif[0x0131] = aime_str # Software
        exif[0xa43b] = aime_str # ImageEditingSoftware
        return exif


    def __init_and_start_manager(self, port_offset=0):
        try:
            APIWorkerInterface.manager = WorkerSyncManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + port_offset), authkey=SYNC_MANAGER_AUTH_KEY)
            if self.rank == 0:
                APIWorkerInterface.barrier = Barrier(self.world_size)
                APIWorkerInterface.manager.start()
        
        except EOFError:
            print(f"Start of SyncManager failed, since the port is already in use. Retrying with different port...")
            self.__init_and_start_manager(port_offset + 1) 


    def __init_manager_and_barrier(self):
        """Register barrier in WorkerSyncManager, initialize WorkerSyncManager and assign them to APIWorkerInterface.barrier and APIWorkerInterface.manager
        """
        WorkerSyncManager.register("barrier", lambda: APIWorkerInterface.barrier)
        WorkerSyncManager.register("error_event", lambda: APIWorkerInterface.error_event)
        self.__init_and_start_manager()

        if not self.rank == 0:
            time.sleep(2)
            APIWorkerInterface.manager.connect()
            APIWorkerInterface.barrier = APIWorkerInterface.manager.barrier()
            APIWorkerInterface.error_event = APIWorkerInterface.manager.error_event()


    def __make_worker_name(self):
        """Make a name for the worker based on worker hostname gpu name and gpu id: 

        Returns:
            str: name of the worker like <hostname>_<gpu_name>_<gpu_id> if gpu_name is given, , else <hostname>_GPU_<gpu_id>
        """        
        worker_name = socket.gethostname()
        for id in range(self.world_size):
            if self.gpu_name:
                worker_name += f'_{self.gpu_name}_{id+self.gpu_id}'
            else:
                worker_name += f'_GPU{id+self.gpu_id}'
        return worker_name


    def __convert_output_types_to_string_representation(
        self,
        output_data,
        output_name,
        output_description,
        job_data
        ):
        """Converts parameters output data from type 'image' and 'image_list' to base64 strings. 

        Args:
            output_data (dict): worker [OUTPUT] parameter dictionary
            finished (bool): Set True fro sending end result, False for progress data

        Returns:
            dict: worker [OUTPUT] parameter dictionary with data converted to base64 string
        """
        if output_name in output_data:
            output_type = output_description.get('type') 
            image_format = output_description.get('format', 'JPEG')
            color_space = output_description.get('color_space', 'RGB')

            if output_type == 'image':
                output_data[output_name] = self.__convert_image_to_base64_string(
                    output_data[output_name],
                    image_format,
                    color_space,
                    job_data
                )
            elif output_type == 'image_list':
                output_data[output_name] = self.__convert_image_list_to_base64_string(
                    output_data[output_name],
                    image_format,
                    color_space,
                    job_data
                )
            elif output_type == 'audio':
                audio_format = output_description.get('format', 'wav')
                output_data[output_name] = self.__convert_audio_to_base64_string(
                    output_data[output_name],
                    audio_format
                )


    def __prepare_output(self, output_data, job_data, finished):
        """Adds SERVER_PARAMETERS to output_data. Converts parameters in output data from type 'image' and 'image_list' to base64 strings. 
        Adds [OUTPUT] parameters found in job_data[output_description/progress_description] to output_data

        Args:
            output_data (dict): worker [OUTPUT] parameter dictionary
            finished (bool): Set True fro sending end result, False for progress data

        Returns:
            dict: worker [OUTPUT] parameter dictionary with data converted to base64 string
        """
        
        if output_data:
            output_data = output_data.copy()
            output_data['version'] = self.version
            output_data['auth'] = self.worker_name
            if finished:
                for parameter in SERVER_PARAMETERS:
                    output_data[parameter] = job_data.get(parameter)
                output_data['job_type'] = self.job_type
                mode = 'final'
            else:
                mode = 'progress'
            descriptions = self.__current_job_cmds.get(job_data.get('job_id'), {}).get(f'{mode}_output_descriptions')
            for output_name, output_description in descriptions.items():
                self.__convert_output_types_to_string_representation(output_data, output_name, output_description, job_data)
                if finished:
                    if output_name in job_data and output_name not in output_data:
                        output_data[output_name] = job_data[output_name]
        return output_data


    def __convert_image_to_base64_string(self, image, image_format, color_space, job_data):
        """Converts given PIL image to base64 string with given image_format and image metadata parsed from job_data.

        Args:
            image (PIL.PngImagePlugin.PngImageFile): Python pillow image to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of image
        """
        image = image.convert(color_space)
        with io.BytesIO() as buffer:
            if image_format == 'PNG':
                image.save(buffer, format=image_format, pnginfo=self.get_pnginfo_metadata(job_data))
            elif image_format =='JPEG' or image_format == 'JPG':
                image_format = 'JPEG'
                exif = self.get_exif_metadata(image, job_data)
                image.save(buffer, format=image_format, exif=exif)
            else:
                image.save(buffer, format=image_format)
            
            image_64 = f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64

    
    def __convert_audio_to_base64_string(self, audio_object, audio_format='wav'):
        return f'data:audio/{audio_format};base64,' + base64.b64encode(audio_object.getvalue()).decode('utf-8')


    def __convert_image_list_to_base64_string(self, list_images, image_format, color_space, job_data):
        """Converts given list of PIL images to base64 string with given image_format and adds image metadata from input data.

        Args:
            list_images (list [PIL.PngImagePlugin.PngImageFile, ..]): List of python pillow images to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of images
        """        
        image_64 = [self.__convert_image_to_base64_string(image, image_format, color_space, job_data) for image in list_images]
        return image_64


    def __print_server_offline_string(self, start_time, dot_string):
        """Prints information about the connection to the API server, if it's offline.

        Args:
            start_time (float): Time since the server is offline
            dot_string (generator): Generator yielding string for dynamic print output f.i. moving dot
        """        
        duration_being_offline = datetime.now() - start_time
        duration_being_offline = duration_being_offline - timedelta(microseconds=duration_being_offline.microseconds)        
        print(f'Connection to API Server {self.api_server} offline for {duration_being_offline}. Trying to reconnect{next(dot_string)}', end='\r')


    def dot_string_generator(self):
        """Generator of string with moving dot for server status print output

        Yields:
            str: '.   ' with dot moving each call
        """
        dot_string = '.   '
        counter = 0
        while True:
            if (counter//3) % 2 == 0:
                dot_string = dot_string[-1] + dot_string[:-1]
            else:
                dot_string = dot_string[1:] + dot_string[0]
            yield dot_string
            counter += 1


    def __print_server_status(self, response):
        """Prints server status to the terminal. Called at start of worker if self.print_server_status is True.

        Args:
            response (requests.models.Response or requests.exceptions.ConnectionError): 

        """        
        if type(response) is requests.models.Response:
            if response.status_code == 200:
                response_json = response.json()
                status = 'online'
                
                if response_json.get('msg'):
                    message_str = f'\nBut server responded with: {response_json.get("cmd")}: {response_json.get("msg")}'
                else:
                    message_str = ''

                if not response_json.get('cmd'):
                    message_str += f'\nUnknown server response: {response_json}\n'
            elif response.status_code == 503:
                status = 'offline'
                message_str = ''

        elif type(response) is requests.exceptions.ConnectionError:
            message_str = ''
            status = 'offline'
        else:
            status = 'unknown'
            message_str = f'\nUnknown server response: {response}\n'

        output_str = \
            '--------------------------------------------------------------\n\n' +\
            f'           API server {self.api_server} {status}' +\
            message_str +\
            '\n\n--------------------------------------------------------------'
        if self.rank == 0:
            print(output_str)


    def __fetch(self, route, json=None, callback=None, error_callback=None):
        """Send post request on given route on API server with given arguments

        Args:
            route (str): Route on API server for post request
            json (dict, optional): Arguments for post request

        Returns:
            requests.models.Response: post request response from API server
            
        Examples:
            Example_response.json() on routes /check_server_status and /worker_job_request: 
                API Server received data without problems:          {'cmd': 'ok'} 
                An error occured in API server:                     {'cmd': 'error', 'msg': <error message>} 
                API Server received data with a warning:   {'cmd': 'warning', 'msg': <warning message>}
        """
        try:
            response = requests.post(self.api_server + route, json=json, timeout=self.request_timeout)
            if callback:
                callback(response)
            return response
        except requests.Timeout as error:
            if error_callback:
                error_callback(error)
            else:
                raise requests.exceptions.ConnectionError
        except Exception as error:
            if error_callback:
                error_callback(error)
            else:
                raise error



    def __fetch_async(
        self,
        route,
        json=None,
        callback=None,
        error_callback=None
        ):
        """Sends non-blocking post request on given route on API server with given arguments. Returns None. Response 
        in __async_fetch_callback or __async_fetch_error_callback.

        Args:
            route (str): Route on API server for post request, f.i. '/worker_job_progress'
            json (dict, optional): Arguments for post request
        """
        if callback:
            self.__custom_callback = callback
        if error_callback:
            self.__custom_error_callback = error_callback
        self.pool_executor.submit(self.__fetch, route, json, self.__async_fetch_callback, self.__async_fetch_error_callback)



    def __async_fetch_callback(self, response):
        """Is called when API server sent a response to __fetch_async. 
        Sets progress_data_received = True and calls __custom_callback, if given to APIWorkerInterface

        Args:
            response (requests.models.Response): Http response from API server.

        Examples:
            Example_response.json() on routes /check_server_status and /worker_job_request:

            .. code-block::

                API Server received data without problems:      {'cmd': 'ok'} 
                An error occured in API server:                 {'cmd': 'error', 'msg': <error message>} 
                API Server received data with a warning:        {'cmd': 'warning', 'msg': <warning message>}
        """
        batch_response = response.json()
        if not isinstance(batch_response, list):
            batch_response = [batch_response]

        for res in batch_response:
            job_id = res.get('job_id')
            if job_id:
                job_cmd = self.__get_current_job_cmd(job_id)
                if job_cmd:
                    progress_input_params = res.get('progress_input_params')
                    if progress_input_params:
                        job_cmd.setdefault('progress_input_params', []).extend(progress_input_params)  
                    canceled = res.get('canceled', False)
                    if canceled and not job_cmd.get('canceled'):
                        job_cmd['canceled'] = True
                    self.__current_job_cmds[job_id] = job_cmd

        self.progress_data_received = True     
        if self.print_server_status:
            self.__print_server_status(response)
            self.print_server_status = False
        if self.__custom_callback:
            self.__custom_callback(response)


    def __async_fetch_error_callback(self, error):
        """Is called when request didn't reach API server is offline. 
        Sets progress_data_received = True and calls progress_received_callback, if given to APIWorkerInterface

        Args:
            error (requests.exceptions.ConnectionError): requests.exceptions.ConnectionError
        """
        if self.print_server_status:
            self.progress_data_received = True
            self.__print_server_status(error)
            self.print_server_status = False
        else:
            if self.__custom_error_callback:
                self.__custom_error_callback(error)


    @staticmethod
    def get_version():
        """Parses name and version of AIME API Worker Interface with pkg_resources

        Returns:
            str: Name and version of AIME API Worker Interface
        """        
        try:
            version = str(pkg_resources.get_distribution('aime_api_worker_interface'))
        except pkg_resources.DistributionNotFound: # If package is not installed via pip
            import re
            from pathlib import Path
            setup_py = Path(__file__).resolve().parent.parent / 'setup.py'
            with open(setup_py, 'r') as file:                
                version_no = re.search(r"version\s*=\s*'(.*)'\s*,\s*\n", file.read()).group(1)
            version = f'AIME API Worker Interface {version_no}'
        return version
