
import time
from datetime import datetime, timedelta

from multiprocessing import Barrier
from multiprocessing.managers import SyncManager
from multiprocessing.dummy import Pool

import requests 
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


class MyManager(SyncManager):    
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
        Example usage simple:

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
        worker_version=0
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
        self.__custom_callback = None
        self.__custom_error_callback = None
        self.image_metadata_params = image_metadata_params
        self.worker_name = self.__make_worker_name()
        self.__init_manager_and_barrier()
        self.progress_data_received = True
        self.__current_job_data = dict()
        self.print_server_status = print_server_status
        self.async_check_server_connection(terminal_output = print_server_status)
        self.request_timeout = request_timeout
        self.worker_version = worker_version
        self.version = self.get_version()


    def job_request(self):
        """Worker requests a job from API Server on route /worker_job_request. If there is 
        no client job offer within the job_timeout = request_timeout * 0.9 the API server responds with 'cmd':'no_job' and the 
        worker requests a job again on route /worker_job_request. 
        In MultGPU-Mode (world_size > 1) only rank 0 will get job_data.

        Returns:
            dict: Job data with worker [INPUT] parameters received from API server.

        Examples:

            Example job data:

            .. highlight:: python
            .. code-block:: python

                job_data = {
                    'prompt': 'prompt',
                    'wait_for_result': False,
                    'job_id': 'JID1',
                    'endpoint_name': 'stable_diffusion_xl_txt2img',
                    'start_time': 1700430052.505548,
                    'start_time_compute': 1700430052.5124364},
                    'cmd': 'job',
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
        job_data = dict()
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
                    'request_timeout': self.request_timeout
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
                    job_data = response.json() 
                    cmd = job_data.get('cmd')
                    msg = job_data.get('msg', 'unknown')
                    if cmd == 'job':
                        have_job = True
                    elif cmd == 'no_job':
                        continue
                    elif cmd == 'error':
                        exit(response_output_str.format(cmd=cmd, msg=msg))
                    elif cmd == 'warning':
                        print(response_output_str.format(cmd=cmd, msg=msg))
                        counter += 1
                        self.check_periodically_if_server_online()
                        if counter > 3:
                            exit(response_output_str.format(cmd=cmd, msg=msg))
                    else:
                        exit(response_output_str.format(cmd='unknown command', msg=cmd))

        if self.world_size > 1:
            # hold all GPU processes here until we have a new job                     
            APIWorkerInterface.barrier.wait()
        self.__current_job_data = job_data

        return job_data


    def send_job_results(self, results, job_data=None):
        """Process/convert job results and send it to API Server on route /worker_job_result.

        Args:
            results (dict): worker [OUTPUT] result parameters (f.i. 'image', 'images' or 'text').
                Example results: ``{'images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}``
            job_data (dict, optional): job_data received from job_request(), if an explicit change of job parameters is desired. 
                Defaults to None.

        Returns:
            requests.models.Response: Http response from API server to the worker.

        Examples:
            Example response.json(): 
            
            .. code-block:: 

                API Server received data without problems:          {'cmd': 'ok'} 
                An error occured in API server:                     {'cmd': 'error', 'msg': <error message>} 
                API Server received data received with a warning:   {'cmd': 'warning', 'msg': <warning message>}
        """
        if job_data:
            self.__current_job_data.update(job_data)
        if self.rank == 0:        
            results = self.__prepare_output(results, True)
            while True:
                try:
                    response =  self.__fetch('/worker_job_result', results)
                except requests.exceptions.ConnectionError:
                    print('Connection to server lost')
                    return
                if response.status_code == 200:
                    return response
                else:
                    self.check_periodically_if_server_online()
                    return


    def send_progress(
        self,
        progress,
        progress_data=None,
        job_data=None,
        progress_received_callback=None,
        progress_error_callback=None
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
            
        """
        
        if job_data: 
            self.__current_job_data.update(job_data) 
        if self.rank == 0 and not self.__current_job_data.pop('wait_for_result', False):
            
            payload = {parameter: self.__current_job_data[parameter] for parameter in SERVER_PARAMETERS}
            payload.update(
                {
                    'progress': progress, 
                    'progress_data': self.__prepare_output(progress_data, False)
                }
            )
            self.progress_data_received = False
            _ = self.__fetch_async('/worker_job_progress', payload, progress_received_callback, progress_error_callback)


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
            dot_string = self.__dot_string_generator()
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


    def get_pnginfo_metadata(self):
        """Parses and returns image metadata from job_data.

        Returns:
            PIL.PngImagePlugin.PngInfo: PngInfo Object with metadata for PNG images
        """        
        metadata = PngInfo()
        for parameter_name in self.image_metadata_params:
            parameter = self.__current_job_data.get(parameter_name)
            if parameter:
                metadata.add_text(parameter_name, str(parameter))
        aime_str = f'AIME API {self.__current_job_data.get("endpoint_name", self.job_type)}'
        metadata.add_text('Artist', aime_str)
        metadata.add_text('ProcessingSoftware', aime_str)
        metadata.add_text('Software', aime_str)
        metadata.add_text('ImageEditingSoftware', aime_str)
        
        return metadata


    def get_exif_metadata(self, image):
        metadata = {str(parameter_name): self.__current_job_data.get(parameter_name) for parameter_name in DEFAULT_IMAGE_METADATA}
        exif = image.getexif()
        exif[0x9286] = json.dumps(metadata) # Comment Tag
        aime_str = f'AIME API {self.__current_job_data.get("endpoint_name", self.job_type)}'
        exif[0x013b] = aime_str # Artist
        exif[0x000b] = aime_str # ProcessingSoftware
        exif[0x0131] = aime_str # Software
        exif[0xa43b] = aime_str # ImageEditingSoftware
        return exif


    def __init_manager_and_barrier(self):
        """Register barrier in MyManager, initialize MyManager and assign them to APIWorkerInterface.barrier and APIWorkerInterface.manager
        """
        if self.world_size > 1:
            MyManager.register("barrier", lambda: APIWorkerInterface.barrier)
            APIWorkerInterface.manager = MyManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + self.gpu_id), authkey=SYNC_MANAGER_AUTH_KEY)
            # multi GPU synchronization required
            if self.rank == 0:
                APIWorkerInterface.barrier = Barrier(self.world_size)
                APIWorkerInterface.manager.start()
            else:
                time.sleep(2)   # manager has to be started first to connect
                APIWorkerInterface.manager.connect()
                APIWorkerInterface.barrier = APIWorkerInterface.manager.barrier()    


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
        output_description
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
            image_format = output_description.get('image_format', 'JPEG')
            color_space = output_description.get('color_space', 'RGB')

            if output_type == 'image':
                output_data[output_name] = self.__convert_image_to_base64_string(
                    output_data[output_name], 
                    image_format, 
                    color_space
                )
            elif output_type == 'image_list':
                output_data[output_name] = self.__convert_image_list_to_base64_string(
                    output_data[output_name], 
                    image_format,
                    color_space
                )


    def __prepare_output(self, output_data, finished):
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
                    output_data[parameter] = self.__current_job_data.get(parameter)
                output_data['job_type'] = self.job_type
                mode = 'output'

            else:
                mode = 'progress'
            descriptions = self.__current_job_data.get(f'{mode}_descriptions')
            for output_name, output_description in descriptions.items():
                self.__convert_output_types_to_string_representation(output_data, output_name, output_description)
                if finished:
                    if output_name in self.__current_job_data and output_name not in output_data:
                        output_data[output_name] = self.__current_job_data[output_name]
        return output_data


    def __convert_image_to_base64_string(self, image, image_format, color_space):
        """Converts given PIL image to base64 string with given image_format and image metadata parsed from __current_job_data.

        Args:
            image (PIL.PngImagePlugin.PngImageFile): Python pillow image to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of image
        """
        image = image.convert(color_space)
        with io.BytesIO() as buffer:
            if image_format == 'PNG':
                image.save(buffer, format=image_format, pnginfo=self.get_pnginfo_metadata())

            elif image_format =='JPEG' or image_format == 'JPG':
                image_format = 'JPEG'
                exif = self.get_exif_metadata(image)
                image.save(buffer, format=image_format, exif=exif)
            else:
                image.save(buffer, format=image_format)
            
            image_64 = f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64


    def __convert_image_list_to_base64_string(self, list_images, image_format, color_space):
        """Converts given list of PIL images to base64 string with given image_format and image metadata parsed from __current_job_data.

        Args:
            list_images (list [PIL.PngImagePlugin.PngImageFile, ..]): List of python pillow images to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of images
        """        
        image_64 = [self.__convert_image_to_base64_string(image, image_format, color_space) for image in list_images]
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


    def __dot_string_generator(self):
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


    def __fetch(self, route, json=None):
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
        except requests.Timeout:
            raise requests.exceptions.ConnectionError

        return response


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
        pool = Pool()
        pool.apply_async(self.__fetch, args=[route, json], callback=self.__async_fetch_callback, error_callback=self.__async_fetch_error_callback)
        pool.close()


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
