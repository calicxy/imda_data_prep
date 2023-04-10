import os
import datasets
from datasets import load_dataset

_DESCRIPTION = """\
This new dataset is designed to solve this great NLP task and is crafted with a lot of care.
"""

_CITATION = """\
"""

_HOMEPAGE = "https://huggingface.co/indonesian-nlp/librivox-indonesia"

_LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"

# TODO: Name of the dataset usually matches the script name with CamelCase instead of snake_case
class NewDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="example", version=VERSION, description="This part of my dataset covers a first domain"),
    ]

    # DEFAULT_CONFIG_NAME = "example"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        
        features = datasets.Features(
            {
                "audio": datasets.features.Audio(sampling_rate=16000),
                "label": datasets.Value("string")
                # These are the features of your dataset like images, labels ...
            }
        )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            supervised_keys=("audio", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        path_to_clips = "./sample_data"
        audio_path = f"{path_to_clips}/waves_yesno.tar.gz"
        local_extracted_archive = dl_manager.extract(audio_path) if not dl_manager.is_streaming else None #how else if we want to stream?
        # metadata_path = dl_manager.download_and_extract(f"{_DATA_URL}/metadata.csv.gz")
        return [
            datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "local_extracted_archive": local_extracted_archive,
                "audio_files": dl_manager.iter_archive(audio_path),
                # "metadata_path": dl_manager.download_and_extract(_METADATA_URL + "/metadata_train.csv.gz"),
                "path_to_clips": path_to_clips,
              },
          ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
            self,
            local_extracted_archive,
            audio_files,
            # metadata_path,
            path_to_clips,
        ):
            """Yields examples."""
            # data_fields = list(self._info().features.keys())
            # metadata = {}
            # with open(metadata_path, "r", encoding="utf-8") as f:
            #     reader = csv.DictReader(f)
            #     for row in reader:
            #         if self.config.name == "_all_" or self.config.name == row["language"]:
            #             row["path"] = os.path.join(path_to_clips, row["path"])
            #             # if data is incomplete, fill with empty values
            #             for field in data_fields:
            #                 if field not in row:
            #                     row[field] = ""
            #             metadata[row["path"]] = row
            id_ = 0
            for path, f in audio_files:
              result = {}
              path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
              result["audio"] = {"path": path, "bytes": f.read()}
              result["label"] = f
              yield id_, result
              id_ += 1
                # if path in metadata:
                #     result = dict(metadata[path])
                #     # set the audio feature and the path to the extracted file
                #     path = os.path.join(local_extracted_archive, path) if local_extracted_archive else path
                #     result["audio"] = {"path": path, "bytes": f.read()}
                #     result["path"] = path
                #     yield id_, result
                #     id_ += 1