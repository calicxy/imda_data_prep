import os
import glob
import datasets
import pandas as pd
from sklearn.model_selection import train_test_split

_DESCRIPTION = """\
The National Speech Corpus (NSC) is the first large-scale Singapore English corpus 
spearheaded by the Info-communications and Media Development Authority (IMDA) of Singapore.
"""

_CITATION = """\
"""
_CHANNEL_CONFIGS = sorted([
    "CHANNEL0", "CHANNEL1", "CHANNEL2"
])

_GENDER_CONFIGS = sorted(["F", "M"])

_RACE_CONFIGS = sorted(["CHINESE", "MALAY", "INDIAN", "OTHERS"])

_HOMEPAGE = "https://www.imda.gov.sg/how-we-can-help/national-speech-corpus"

_LICENSE = ""

_PATH_TO_DATA = r'C:\Users\calic\Downloads\huggingface-dataset\imda-dataset\IMDA - National Speech Corpus\PART1'
# _PATH_TO_DATA = './PART1/DATA'

class Minds14Config(datasets.BuilderConfig):
    """BuilderConfig for xtreme-s"""

    def __init__(
        self, channel, gender, race, description, homepage, path_to_data
    ):
        super(Minds14Config, self).__init__(
            name=channel+gender+race,
            version=datasets.Version("1.0.0", ""),
            description=self.description,
        )
        self.channel = channel
        self.gender = gender
        self.race = race
        self.description = description
        self.homepage = homepage
        self.path_to_data = path_to_data


def _build_config(channel, gender, race):
    return Minds14Config(
        channel=channel,
        gender=gender,
        race=race,
        description=_DESCRIPTION,
        homepage=_HOMEPAGE,
        path_to_data=_PATH_TO_DATA,
    )

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
    BUILDER_CONFIGS = []
    for channel in _CHANNEL_CONFIGS + ["all"]:
        for gender in _GENDER_CONFIGS + ["all"]:
            for race in _RACE_CONFIGS + ["all"]:
                BUILDER_CONFIGS.append(_build_config(channel, gender, race))
    # BUILDER_CONFIGS = [_build_config(name) for name in _CHANNEL_CONFIGS + ["all"]]

    DEFAULT_CONFIG_NAME = "allallall"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        task_templates = None
        # mics = _CHANNEL_CONFIGS
        features = datasets.Features(
            {
                "audio": datasets.features.Audio(sampling_rate=16000),
                "transcript": datasets.Value("string"),
                "mic": datasets.Value("string"),
                "audio_name": datasets.Value("string"),
                "gender": datasets.Value("string"),
                "race": datasets.Value("string"),
            }
        )
        
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            supervised_keys=("audio", "transcript"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
            task_templates=task_templates,
        )

    def _split_generators(self, dl_manager):
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name
        mics = (
            _CHANNEL_CONFIGS
            if self.config.channel == "all"
            else [self.config.channel]
        )

        gender = (
            _GENDER_CONFIGS
            if self.config.gender == "all"
            else [self.config.gender]
        )

        race = (
            _RACE_CONFIGS
            if self.config.race == "all"
            else [self.config.race]
        )

        # augment speaker ids directly here
        # read the speaker information
        train_speaker_ids = []
        test_speaker_ids = []
        path_to_speaker = os.path.join(self.config.path_to_data, "DOC", "Speaker Information (Part 1).XLSX")
        speaker_df = pd.read_excel(path_to_speaker, dtype={'SCD/PART1': object})
        for g in gender:
            for r in race:
                X = speaker_df[(speaker_df["ACC"]==r) & (speaker_df["SEX"]==g)]
                X_train, X_test = train_test_split(X, test_size=0.3, random_state=42, shuffle=True)
                train_speaker_ids.extend(X_train["SCD/PART1"])
                test_speaker_ids.extend(X_test["SCD/PART1"])

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        return [
            datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "path_to_data": self.config.path_to_data,
                "speaker_metadata":speaker_df,
                # "speaker_ids": train_speaker_ids,
                "speaker_ids":["0001"],
                "mics": mics,
                "dl_manager": dl_manager
              },
          ),
          datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                "path_to_data": self.config.path_to_data,
                "speaker_metadata":speaker_df,
                # "speaker_ids": test_speaker_ids,
                "speaker_ids": ["0003"],
                "mics": mics,
                "dl_manager": dl_manager
            },
        ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
            self,
            path_to_data,
            speaker_metadata,
            speaker_ids,
            mics,
            dl_manager
        ):
        id_ = 0
        for mic in mics:
            for speaker in speaker_ids:
                # TRANSCRIPT: in the case of error, if no file found then dictionary will b empty
                metadata_path = os.path.join(path_to_data, "DATA", mic, "SCRIPT", mic[-1]+speaker+'*.TXT')
                script_list = glob.glob(metadata_path)
                d = {}
                for script in script_list:
                    line_num = 0
                    with open(script, encoding='utf-8-sig') as f:
                        for line in f:
                            if line_num == 0:
                                key = line.split("\t")[0]
                                line_num += 1
                            elif line_num == 1:
                                d[key] = line.strip()
                                line_num -= 1
                # AUDIO: in the case of error it will skip the speaker
                archive_path = os.path.join(path_to_data, "DATA", mic, "WAVE", "SPEAKER"+speaker+'.zip')
                # check that archive path exists, else will not open the archive
                if os.path.exists(archive_path):
                    audio_files = dl_manager.iter_archive(archive_path)
                    for path, f in audio_files:
                        # bug catching if any error?
                        result = {}
                        full_path = os.path.join(archive_path, path) if archive_path else path # bug catching here
                        result["audio"] = {"path": full_path, "bytes": f.read()}
                        result["transcript"] = d[f.name[-13:-4]]
                        result["audio_name"] = path
                        result["mic"] = mic
                        metadata_row = speaker_metadata.loc[speaker_metadata["SCD/PART1"]==speaker].iloc[0]
                        result["gender"]=metadata_row["SEX"]
                        result["race"]=metadata_row["ACC"]
                        yield id_, result
                        id_ += 1