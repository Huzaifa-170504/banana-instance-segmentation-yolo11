
from pathlib import Path
import shutil
import random
import json
from collections import defaultdict


class DatasetPreparer:

    def __init__(self, root):

        self.root = Path(root)

        self.raw = (
            self.root /
            "01_Datasets" /
            "dataset"
        )

        self.output = (
            self.root /
            "01_Datasets" /
            "Prepared_Dataset"
        )

        self.logs = (
            self.root /
            "05_Logs"
        )

        self.logs.mkdir(exist_ok=True)

        self.corrections = {
            "files_corrected":0,
            "coordinates_clipped":0
        }


    def clean_label(self, source, destination):

        changed=False
        new_lines=[]

        with open(source) as f:

            lines=f.readlines()


        for line in lines:

            parts=line.strip().split()

            if not parts:
                continue

            cls=parts[0]

            coords=[]

            for value in parts[1:]:

                v=float(value)

                if v < 0:

                    v=0
                    self.corrections["coordinates_clipped"]+=1
                    changed=True


                if v > 1:

                    v=1
                    self.corrections["coordinates_clipped"]+=1
                    changed=True


                coords.append(str(v))


            new_lines.append(
                cls+" "+" ".join(coords)
            )


        if changed:
            self.corrections["files_corrected"]+=1


        destination.write_text(
            "\n".join(new_lines)
        )



    def collect_files(self):

        files=[]

        for img in self.raw.rglob("*.jpg"):

            label=img.with_suffix(".txt")

            if label.exists():

                files.append(
                    (img,label)
                )

        return files



    def create_structure(self,name):

        base=self.output/name

        for folder in [
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test"
        ]:

            (base/folder).mkdir(
                parents=True,
                exist_ok=True
            )


    def copy_set(self,data,name):

        base=self.output/name

        for split,items in data.items():

            for img,label in items:

                shutil.copy2(
                    img,
                    base/"images"/split/img.name
                )

                self.clean_label(
                    label,
                    base/"labels"/split/label.name
                )


    def yaml_file(self,name):

        text=f"""
path: {self.output/name}

train: images/train
val: images/val
test: images/test

names:
  0: healthy_banana
  1: unhealthy_banana
"""

        (self.output/name/"dataset.yaml").write_text(text)



    def prepare(self):

        files=self.collect_files()

        print("Total pairs:",len(files))


        # benchmark random split

        random.seed(42)

        random.shuffle(files)

        n=len(files)

        benchmark={

            "train":files[:int(n*.8)],

            "val":files[int(n*.8):int(n*.9)],

            "test":files[int(n*.9):]

        }


        self.create_structure("benchmark")

        self.copy_set(
            benchmark,
            "benchmark"
        )

        self.yaml_file(
            "benchmark"
        )



        # research split by family

        families=defaultdict(list)

        for img,label in files:

            family=img.stem.split("_rot")[0]

            families[family].append(
                (img,label)
            )


        keys=list(families.keys())

        random.shuffle(keys)


        research={

            "train":[],
            "val":[],
            "test":[]

        }


        for i,key in enumerate(keys):

            if i < 5:
                split="train"

            elif i < 7:
                split="val"

            else:
                split="test"


            research[split]+=families[key]


        self.create_structure("research")


        self.copy_set(
            research,
            "research"
        )

        self.yaml_file(
            "research"
        )


        with open(
            self.logs/"label_correction_report.json",
            "w"
        ) as f:

            json.dump(
                self.corrections,
                f,
                indent=4
            )


        with open(
            self.logs/"dataset_preparation_report.json",
            "w"
        ) as f:

            json.dump(
                {
                "total_images":len(files),
                "benchmark_split":{
                    k:len(v)
                    for k,v in benchmark.items()
                },
                "research_split":{
                    k:len(v)
                    for k,v in research.items()
                }
                },
                f,
                indent=4
            )


        print("Preparation complete")

