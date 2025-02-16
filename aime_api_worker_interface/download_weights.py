import argparse
from getpass import getpass
from pathlib import Path
import sys


try:
    from huggingface_hub import snapshot_download, login, whoami
    from huggingface_hub.errors import GatedRepoError, HFValidationError, LocalEntryNotFoundError, LocalTokenNotFoundError, RepositoryNotFoundError
    from requests.exceptions import HTTPError
    HF_HUB_PRESENT = True

except ModuleNotFoundError:
    HF_HUB_PRESENT = False


AIME_HOSTED_MODELS = []


class ModelDownloader:

    def __init__(self):
        self.args = self.load_flags()
        self.prepare_download_dir()
        self.user_info = self.get_user_info()


    def load_flags(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'model', type=str,
            help='Model name to download'
        )
        parser.add_argument(
            '-o', '--download_dir', type=str,
            help='Download directory'
        )
        parser.add_argument(
            '-wo', '--max_workers', type=int, default=8,
            help='Maximum number of workers for downloading'
        )
        parser.add_argument(
            '-or', '--include_original', action='store_true',
            help='Include folder with original weights',
        )
        args = parser.parse_args()
        args.model = args.model.strip('/')
        return args


    def prepare_download_dir(self):
        download_dir = Path(self.args.download_dir or Path.cwd())
        if download_dir.is_dir():
            model_name = self.args.model.split('/')[-1]
            download_dir = download_dir / model_name
            if not download_dir.is_dir():
                download_dir.mkdir()
            self.args.download_dir = download_dir
        else:
            exit(f'Invalid path {download_dir}')


    def start_download(self):
        if self.args.model in AIME_HOSTED_MODELS:
            self.download_from_aime()
        else:
            if HF_HUB_PRESENT:
                self.download_from_hf()
            else:
                exit("You're trying to download from huggingface, but the pip package huggingface_hub is missing! Install it with:\n\npip install huggingface_hub")


    def download_from_aime(self):
        pass


    def download_from_hf(self):
        try:
            self.start_hf_download()
        except GatedRepoError:
            if self.user_info:
                print(
                    f'This account has no access to the model {self.args.model}. '
                    f'Visit https://huggingface.co/{self.args.model} to ask for '
                    f'access or enter another access token (Hidden): ',
                    end='',
                    flush=True
                )
            else:
                print(
                    f'Access to model {self.args.model} is restricted. '
                    f'Please enter your Huggingface access token (Hidden): ',
                    end='',
                    flush=True
                )
            self.hf_login()
            try:
                self.start_hf_download()
            except GatedRepoError as error:
                exit(
                    f'Access to model {self.args.model} is not permitted on this account. '
                    f'Visit https://huggingface.co/{self.args.model} to ask for access.'
                )
        except (HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError) as error:
            exit(error)


    def hf_login(self):
        hf_token = getpass('')
        if hf_token:
            try:
                login(hf_token)
                self.user_info = whoami()  # Checks if the user is logged in
                print(f'Logged in as {self.user_info.get("name")}')
            except HTTPError as error:
                exit(error)
        else:
            exit('No access token given! Closing...')


    def get_user_info(self):
        if HF_HUB_PRESENT:
            try:
                user_info = whoami()  # Checks if the user is logged in
                print(f'Logged in as {user_info.get("name")}')
                return user_info
            except LocalTokenNotFoundError:
                print('Not logged in to Hugging face!')


    def start_hf_download(self):
        print(f'Start download from https://huggingface.co/{self.args.model}:')
        snapshot_download(
            repo_id=self.args.model,
            max_workers=self.args.max_workers,
            local_dir=self.args.download_dir,
            ignore_patterns='original/*' if not self.args.include_original else None
        )
        sys.stdout.flush()
        print('\nDownload Complete')



def main():
    model_downloader = ModelDownloader()
    model_downloader.start_download()


if __name__ == "__main__":
    main()
