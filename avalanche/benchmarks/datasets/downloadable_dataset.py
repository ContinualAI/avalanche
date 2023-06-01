################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 25-05-2021                                                             #
# Author: Lorenzo Pellegrini                                                   #
# E-mail: contact@continualai.org                                              #
# Website: www.continualai.org                                                 #
################################################################################

from abc import abstractmethod, ABC
from pathlib import Path
from typing import TypeVar, Union, Optional

import shutil

import os
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    extract_archive,
    download_url,
    check_integrity,
)

from avalanche.benchmarks.datasets.dataset_utils import default_dataset_location

T_co = TypeVar("T_co", covariant=True)


class DownloadableDataset(Dataset[T_co], ABC):
    """Base class for a downloadable dataset.

    It is recommended to extend this class if a dataset can be downloaded from
    the internet. This implementation codes the recommended behaviour for
    downloading and verifying the dataset.

    The dataset child class must implement the `_download_dataset`,
    `_load_metadata` and `_download_error_message` methods

    The child class, in its constructor, must call the already implemented
    `_load_dataset` method (otherwise nothing will happen).

    A further simplification can be obtained by extending
    :class:`SimpleDownloadableDataset` instead of this class.
    :class:`SimpleDownloadableDataset` is recommended if a single archive is to
    be downloaded and extracted to the root folder "as is".

    The standardized procedure coded by `_load_dataset` is as follows:

    - First, `_load_metadata` is called to check if the dataset can be correctly
      loaded at the `root` path. This method must check if the data found
      at the `root` path is correct and that metadata can be correctly loaded.
      If this method succeeds (by returning True) the process is completed.
    - If `_load_metadata` fails (by returning False or by raising an error),
      then a download will be attempted if the download flag was set to True.
      The download must be implemented in `_download_dataset`. The
      procedure can be drastically simplified by using the `_download_file`,
      `_extract_archive` and `_download_and_extract_archive` helpers.
    - If the download succeeds (doesn't raise an error), then `_load_metadata`
      will be called again.

    If an error occurs, the `_download_error_message` will be called to obtain
    a message (a string) to show to the user. That message should contain
    instructions on how to download and prepare the dataset manually.
    """

    def __init__(
        self,
        root: Union[str, Path],
        download: bool = True,
        verbose: bool = False,
    ):
        """Creates an instance of a downloadable dataset.

        Consider looking at the class documentation for the precise details on
        how to extend this class.

        Beware that calling this constructor only fills the `root` field. The
        download and metadata loading procedures are triggered only by
        calling `_load_dataset`.

        :param root: The root path where the dataset will be downloaded.
            Consider passing a path obtained by calling
            `default_dataset_location` with the name of the dataset.
        :param download: If True, the dataset will be downloaded if needed.
            If False and the dataset can't be loaded from the provided root
            path, an error will be raised when calling the `_load_dataset`
            method. Defaults to True.
        :param verbose: If True, some info about the download process will be
            printed. Defaults to False.
        """

        super(DownloadableDataset, self).__init__()
        self.root: Path = Path(root)
        """
        The path to the dataset.
        """

        self.download: bool = download
        """
        If True, the dataset will be downloaded (only if needed).
        """

        self.verbose: bool = verbose
        """
        If True, some info about the download process will be printed.
        """

    def _load_dataset(self) -> None:
        """
        The standardized dataset download and load procedure.

        For more details on the coded procedure see the class documentation.

        This method shouldn't be overridden.

        This method will raise and error if the dataset couldn't be loaded
        or downloaded.

        :return: None
        """
        metadata_loaded = False
        metadata_load_error = None
        try:
            metadata_loaded = self._load_metadata()
        except Exception as e:
            metadata_load_error = e

        if metadata_loaded:
            if self.verbose:
                print("Files already downloaded and verified")
            return

        if not self.download:
            msg = (
                "Error loading dataset metadata (dataset download was "
                'not attempted as "download" is set to False)'
            )
            if metadata_load_error is None:
                raise RuntimeError(msg)
            else:
                print(msg)
                raise metadata_load_error

        try:
            self._download_dataset()
        except Exception as e:
            err_msg = self._download_error_message()
            print(err_msg, flush=True)
            raise e

        if not self._load_metadata():
            err_msg = self._download_error_message()
            print(err_msg)
            raise RuntimeError(
                "Error loading dataset metadata (... but the download "
                "procedure completed successfully)"
            )

    @abstractmethod
    def _download_dataset(self) -> None:
        """
        The download procedure.

        This procedure is called only if `_load_metadata` fails.

        This method must raise an error if the dataset can't be downloaded.

        Hints: don't re-invent the wheel! There are ready-to-use helper methods
        like `_download_and_extract_archive`, `_download_file` and
        `_extract_archive` that can be used.

        :return: None
        """
        pass

    @abstractmethod
    def _load_metadata(self) -> bool:
        """
        The dataset metadata loading procedure.

        This procedure is called at least once to load the dataset metadata.

        This procedure should return False if the dataset is corrupted or if it
        can't be loaded.

        :return: True if the dataset is not corrupted and could be successfully
        loaded.
        """
        pass

    @abstractmethod
    def _download_error_message(self) -> str:
        """
        Returns the error message hinting the user on how to download the
        dataset manually.

        :return: A string representing the message to show to the user.
        """
        pass

    def _cleanup_dataset_root(self):
        """
        Utility method that can be used to remove the dataset root directory.

        Can be useful if a cleanup is needed when downloading and extracting the
        dataset.

        This method will also re-create the root directory.

        :return: None
        """
        shutil.rmtree(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _download_file(self, url: str, file_name: str, checksum: Optional[str]) -> Path:
        """
        Utility method that can be used to download and verify a file.

        :param url: The download url.
        :param file_name: The name of the file to save. The file will be saved
            in the `root` with this name. Always fill this parameter.
            Don't pass a path! Pass a file name only!
        :param checksum: The MD5 hash to use when verifying the downloaded
            file. Can be None, in which case the check will be skipped.
            It is recommended to always fill this parameter.
        :return: The path to the downloaded file.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        download_url(url, str(self.root), filename=file_name, md5=checksum)
        return self.root / file_name

    def _extract_archive(
        self,
        path: Union[str, Path],
        sub_directory: Optional[str] = None,
        remove_archive: bool = False,
    ) -> Path:
        """
        Utility method that can be used to extract an archive.

        :param path: The complete path to the archive (for instance obtained
            by calling `_download_file`).
        :param sub_directory: The name of the sub directory where to extract the
            archive. Can be None, which means that the archive will be extracted
            in the root. Beware that some archives already have a root directory
            inside of them, in which case it's probably better to use None here.
            Defaults to None.
        :param remove_archive: If True, the archive will be deleted after a
            successful extraction. Defaults to False.
        :return:
        """

        if sub_directory is None:
            extract_root = self.root
        else:
            extract_root = self.root / sub_directory

        extract_archive(
            str(path), to_path=str(extract_root), remove_finished=remove_archive
        )

        return extract_root

    def _download_and_extract_archive(
        self,
        url: str,
        file_name: str,
        checksum: Optional[str],
        sub_directory: Optional[str] = None,
        remove_archive: bool = False,
    ) -> Path:
        """
        Utility that downloads and extracts an archive.

        :param url: The download url.
        :param file_name: The name of the archive. The file will be saved
            in the `root` with this name. Always fill this parameter.
            Don't pass a path! Pass a file name only!
        :param checksum: The MD5 hash to use when verifying the downloaded
            archive. Can be None, in which case the check will be skipped.
            It is recommended to always fill this parameter.
        :param sub_directory: The name of the sub directory where to extract the
            archive. Can be None, which means that the archive will be extracted
            in the root. Beware that some archives already have a root directory
            inside of them, in which case it's probably better to use None here.
            Defaults to None.
        :param remove_archive: If True, the archive will be deleted after a
            successful extraction. Defaults to False.
        :return: The path to the extracted archive. If `sub_directory` is None,
            then this will be the `root` path.
        """
        if sub_directory is None:
            extract_root = self.root
        else:
            extract_root = self.root / sub_directory

        self.root.mkdir(parents=True, exist_ok=True)
        try:
            download_and_extract_archive(
                url,
                str(self.root),
                extract_root=str(extract_root),
                filename=file_name,
                md5=checksum,
                remove_finished=remove_archive,
            )
        except BaseException:
            print(
                "Error while downloading the dataset archive. "
                "The partially downloaded archive will be removed."
            )
            attempt_fpath = self.root / file_name
            attempt_fpath.unlink(missing_ok=True)
            raise

        return extract_root

    def _check_file(self, path: Union[str, Path], checksum: str) -> bool:
        """
        Utility method to check a file.

        :param path: The path to the file.
        :param checksum: The MD5 hash to use.
        :return: True if the MD5 hash of the file matched the given one.
        """
        return check_integrity(str(path), md5=checksum)


class SimpleDownloadableDataset(DownloadableDataset[T_co], ABC):
    """
    Base class for a downloadable dataset consisting of a single archive file.

    It is recommended to extend this class if a dataset can be downloaded from
    the internet as a single archive. For multi-file implementation or if
    a more fine-grained control is required, consider extending
    :class:`DownloadableDataset` instead.

    This is a simplified version of :class:`DownloadableDataset` where the
    following assumptions must hold:
    - The dataset is made of a single archive.
    - The archive must be extracted to the root folder "as is" (which means
        that no subdirectories must be created).

    The child class is only required to extend the `_load_metadata` method,
    which must check the dataset integrity and load the dataset metadata.

    Apart from that, the same assumptions of :class:`DownloadableDataset` hold.
    Remember to call the `_load_dataset` method in the child class constructor.
    """

    def __init__(
        self,
        root_or_dataset_name: Union[str, Path],
        url: str,
        checksum: Optional[str],
        download: bool = False,
        verbose: bool = False,
    ):
        """
        Creates an instance of a simple downloadable dataset.

        Consider looking at the class documentation for the precise details on
        how to extend this class.

        Beware that calling this constructor only fills the `root` field. The
        download and metadata loading procedures are triggered only by
        calling `_load_dataset`.

        :param root_or_dataset_name: The root path where the dataset will be
            downloaded. If a directory name is passed, then the root obtained by
            calling `default_dataset_location` will be used (recommended).
            To check if this parameter is a path, the constructor will check if
            it contains the '\' or '/' characters or if it is a Path instance.
        :param url: The url of the archive.
        :param checksum: The MD5 hash to use when verifying the downloaded
            archive. Can be None, in which case the check will be skipped.
            It is recommended to always fill this parameter.
        :param download: If True, the dataset will be downloaded if needed.
            If False and the dataset can't be loaded from the provided root
            path, an error will be raised when calling the `_load_dataset`
            method. Defaults to False.
        :param verbose: If True, some info about the download process will be
            printed. Defaults to False.
        """

        self.url = url
        self.checksum = checksum

        if (
            isinstance(root_or_dataset_name, Path)
            or "/" in root_or_dataset_name
            or "\\" in root_or_dataset_name
        ):
            root = Path(root_or_dataset_name)
        else:
            root = default_dataset_location(root_or_dataset_name)

        super(SimpleDownloadableDataset, self).__init__(
            root, download=download, verbose=verbose
        )

    def _download_dataset(self) -> None:
        filename = os.path.basename(self.url)
        self._download_and_extract_archive(
            self.url,
            filename,
            self.checksum,
            sub_directory=None,
            remove_archive=False,
        )

    def _download_error_message(self) -> str:
        return (
            "Error downloading the dataset. Consider downloading "
            "it manually at: " + self.url + " and placing it "
            "in: " + str(self.root)
        )


__all__ = ["DownloadableDataset", "SimpleDownloadableDataset"]
