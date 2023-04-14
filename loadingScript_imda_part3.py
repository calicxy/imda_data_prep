import os
import datasets
# import pandas as pd
from sklearn.model_selection import train_test_split
from textgrid import textgrid
import soundfile as sf
from clean_transcript import cleanup_string

_DESCRIPTION = """\
The National Speech Corpus (NSC) is the first large-scale Singapore English corpus 
spearheaded by the Info-communications and Media Development Authority (IMDA) of Singapore.
"""

_CITATION = """\
"""
_CHANNEL_CONFIGS = sorted([
    "Audio Same CloseMic", "Audio Separate IVR", "Audio Separate StandingMic"
])

_HOMEPAGE = "https://www.imda.gov.sg/how-we-can-help/national-speech-corpus"

_LICENSE = ""

_PATH_TO_DATA = r'C:\Users\calic\Downloads\huggingface-dataset\imda-dataset\IMDA - National Speech Corpus\PART3'
# _PATH_TO_DATA = './PART1/DATA'

INTERVAL_MAX_LENGTH = 25

class Minds14Config(datasets.BuilderConfig):
    """BuilderConfig for xtreme-s"""

    def __init__(
        self, channel, description, homepage, path_to_data
    ):
        super(Minds14Config, self).__init__(
            name=channel,
            version=datasets.Version("1.0.0", ""),
            description=self.description,
        )
        self.channel = channel
        self.description = description
        self.homepage = homepage
        self.path_to_data = path_to_data


def _build_config(channel):
    return Minds14Config(
        channel=channel,
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
        BUILDER_CONFIGS.append(_build_config(channel))
    # BUILDER_CONFIGS = [_build_config(name) for name in _CHANNEL_CONFIGS + ["all"]]

    DEFAULT_CONFIG_NAME = "all"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        # TODO: This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        task_templates = None
        features = datasets.Features(
            {
                "audio": datasets.features.Audio(sampling_rate=16000),
                "transcript": datasets.Value("string"),
                "mic": datasets.Value("string"),
                "audio_name": datasets.Value("string"),
                "interval": datasets.Value("string")
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

        audio_list = []
        for mic in mics:
            for (root,dirs,files) in os.walk(os.path.join(self.config.path_to_data, mic), topdown=True):
                if len(files) != 0:
                    for file in files:
                        # get audio path
                        audio_path = os.path.join(root, file)
                        audio_list.append(audio_path)


        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive
        return [
            datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                # "path_to_data": os.path.join(self.config.path_to_data, "Audio Same CloseMic"),
                "audio_list":audio_list,
                "mics": mics,
              },
          ),
          datasets.SplitGenerator(
            name=datasets.Split.TEST,
            gen_kwargs={
                # "path_to_data": os.path.join(self.config.path_to_data, "Audio Same CloseMic"),
                "audio_list": audio_list,
                "mics": mics,
            },
        ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(
            self,
            audio_list,
            mics,
        ):
        id_ = 0
        for audio_path in audio_list:
            file = os.path.split(audio_path)[-1]
            folder = os.path.split(os.path.split(audio_path)[0])[-1]

            # get script_path
            if folder.split("_")[0] == "conf":
                # mic == "Audio Separate IVR"
                script_path = os.path.join(self.config.path_to_data, "Scripts Separate", folder+"_"+file[:-4]+".TextGrid")
            elif folder.split()[1] == "Same":
                # mic == "Audio Same CloseMic IVR"
                script_path = os.path.join(self.config.path_to_data, "Scripts Same", file[:-4]+".TextGrid")
            elif folder.split()[1] == "Separate":
                # mic == "Audio Separate StandingMic":
                script_path = os.path.join(self.config.path_to_data, "Scripts Separate", file[:-4]+".TextGrid")
            

            # LOAD TRANSCRIPT
            # script_path = os.path.join(self.config.path_to_data, 'Scripts Same', '3000-1.TextGrid')
            # check that the textgrid file can be read
            try:
                tg = textgrid.TextGrid.fromFile(script_path)
            except:
                continue
            # LOAD AUDIO
            # archive_path = os.path.join(path_to_data, '3000-1.wav')
            # check that archive path exists, else will not open the archive
            if os.path.exists(audio_path):
                # read into a numpy array using soundfile
                data, sr = sf.read(audio_path)
                result = {}
                i = 0
                intervalLength = 0
                intervalStart = 0
                transcript_list = []
                filepath = os.path.join(self.config.path_to_data, 'tmp_clip.wav')
                while i < (len(tg[0])-1):
                    transcript = cleanup_string(tg[0][i].mark)
                    if intervalLength == 0 and len(transcript) == 0:
                        intervalStart = tg[0][i].maxTime
                        i+=1
                        continue
                    intervalLength += tg[0][i].maxTime-tg[0][i].minTime
                    if intervalLength > INTERVAL_MAX_LENGTH:
                        print(f"INTERVAL LONGER THAN {intervalLength}")
                        result["transcript"] = transcript
                        result["interval"] = "start:"+str(tg[0][i].minTime)+", end:"+str(tg[0][i].maxTime)
                        result["audio"] = {"path": audio_path, "bytes": data[int(tg[0][i].minTime*sr):int(tg[0][i].maxTime*sr)], "sampling_rate":sr}
                        yield id_, result
                        id_+= 1
                        intervalLength = 0
                    else:
                        if (intervalLength + tg[0][i+1].maxTime-tg[0][i+1].minTime) < INTERVAL_MAX_LENGTH:
                            if len(transcript) != 0:
                                transcript_list.append(transcript)
                            i+=1
                            continue
                        if len(transcript) == 0:
                            spliced_audio = data[int(intervalStart*sr):int(tg[0][i].minTime*sr)]
                        else:
                            transcript_list.append(transcript)
                            spliced_audio = data[int(intervalStart*sr):int(tg[0][i].maxTime*sr)]
                        sf.write(filepath, spliced_audio, sr)
                        result["interval"] = "start:"+str(intervalStart)+", end:"+str(tg[0][i].maxTime)
                        result["audio"] = {"path": filepath, "bytes": spliced_audio, "sampling_rate":sr}
                        result["transcript"] = ' '.join(transcript_list)
                        yield id_, result
                        id_+= 1
                        intervalLength=0
                        intervalStart=tg[0][i].maxTime
                        transcript_list = []
                    i+=1
                    

            # audio_files = dl_manager.iter_archive(archive_path)
            # for path, f in audio_files:
            #     # bug catching if any error?
            #     result = {}
            #     full_path = os.path.join(archive_path, path) if archive_path else path # bug catching here
            #     result["audio"] = {"path": full_path, "bytes": f.read()}
            #     result["transcript"] = "placeholder"
            #     result["audio_name"] = path
            #     result["mic"] = mic
            #     yield id_, result
            #     id_ += 1