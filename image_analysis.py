import copy
import cv2
import glob
from itertools import permutations
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import torch

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

print("\ntorch.version:", torch.version, "\n")
print("torch.version.cuda:", torch.version.cuda, "\n")
print("torch.cuda.is_available():", torch.cuda.is_available(), "\n")
print("torch._C._cuda_getDeviceCount():", torch._C._cuda_getDeviceCount(), "\n")

# a collection of functions to create common model architectures listed in
# MODEL_ZOO.md, and optionally load their pre-trained weights
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

class Image_Analyzer:
  """
  A class for simplified use of detectron2's predictor, tailored to the specific
  needs of the SOCMINTEX project.
  """

  def __init__(self, model:str="COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"):
    """
    Initializes the model, this step is only required once per session
    of analysis. Whilst the model is technically up to the user, all the
    following methods are tailored for the default one.

    Parameters:
    -----------
    model : str, optional
      The desired model to be initialized, by default is tailored for
      panoptic segmentation.
    """
    # The parameters are locked behind private access, to limit unintended use
    self.__cfg = get_cfg()
    self.__cfg.merge_from_file(model_zoo.get_config_file(model))
    self.__cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
    self.__predictor = DefaultPredictor(self.__cfg)

  def get_predictor(self):
    """
    This is a getter function for the predictor.
    """
    return self.__predictor

  def analyze(self, img, return_display:bool=False, other_functions:list=[]):
    """
    Takes an image, performs panoptic segmentation on it, and returns a list
    of identified objects within it. For each object in the list, if it is
    a thing (cf. COCO), the description contains 3 fields:
    'category': the name, in English, of the thing.
    'confidence': its confidence score.
    'area': the portion of the image that the object takes up, from 0 to 1.
    If instead, the object is stuff (cf. COCO), meaning an uncountable
    substance, then it has the same parameters as a thing, minus the
    confidence score.

    WARNING: This method relies on the default panoptic segmentation model, if
    you are using a different model, it might not work.

    Parameters:
    -----------
    img : numpy.ndarray
      The image to be analyzed.
    return_display : bool, optional
      Used to request an annotation of the image.
    """
    # Call on detectron2's basic predictor to perform the analysis
    panoptic_seg, segments_info = self.__predictor(img)["panoptic_seg"]

    # Determine the proportions of each class present in the image matrix
    panoptic_seg_np = panoptic_seg.detach().cpu().numpy()
    breakdown = {}
    for i in range(1, len(segments_info)+1):
      breakdown[i] = np.count_nonzero(panoptic_seg_np==i)
    img_area = img.shape[0]*img.shape[1]

    # Create the new list of object descriptions using basic detectron2 results
    results = []
    for i in range(len(segments_info)):
      # Only 'thing' classes have 'instance_id'
      if 'instance_id' in segments_info[i].keys():
        thing = {'category':0, 'area':0, 'confidence':0}
        thing['category'] = MetadataCatalog.get(self.__cfg.DATASETS.TRAIN[0]).thing_classes[segments_info[i]['category_id']]
        thing['confidence'] = segments_info[i]['score']
        thing['area'] = breakdown[segments_info[i]['id']]/img_area
        results.append(thing)
      else:
        stuff = {'category':0, 'area':0}
        stuff['category'] = MetadataCatalog.get(self.__cfg.DATASETS.TRAIN[0]).stuff_classes[segments_info[i]['category_id']]
        stuff['area'] = breakdown[segments_info[i]['id']]/img_area
        results.append(stuff)
    unidentified_area = 1 - (sum(breakdown.values())/img_area)
    results.append({'category':'unidentified', 'area': unidentified_area})

    # An optional feature to perform further analysis on the image
    if other_functions!=[]:
      for function in other_functions:
        # Results must be presented in a way to be added to the results list
        result = function(img)
        results.append(result)

    if not return_display:
      return results

    # Draw the segmentation masks on a copy of the image and return it
    else:
      v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(self.__cfg.DATASETS.TRAIN[0]), scale=1.2)
      out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
      annotated_img = out.get_image()[:, :, ::-1]
      resized_annotation = cv2.resize(annotated_img, img.shape[1::-1])
      comparison = cv2.hconcat([img, resized_annotation])
      # The percentage of unidentified area is written in purple on the image
      cv2.putText(comparison, str(round(100*unidentified_area, 2))+'%', (int(comparison.shape[1]/2), 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)
      return results, comparison

  def analyze_multiple(self, directory0:str, n:int, file_path, directory1:str="", ret:bool=False):
    """
    Runs the analyze() method on n images present in the
    directory. The results are then converted to strings and printed into a
    .txt document at the given file_path. This file_path must include the name
    of the future .txt file, not just its destination directory. If a
    directory1 is provided, it means that the annotated images are expected to
    be created and saved at this second location. If ret (short for return) is
    made True, then the results will also be returned as a Python list.

    WARNING: This method relies on the default panoptic segmentation model, if
    you are using a different model, it might not work.

    Parameters:
    -----------
    directory0 : str
      The path of the directory containing the collection of images
      to be analyzed.
    n : int
      The number of images to be analyzed from the directory. If it
      surpasses the size of the directory, then the entire directory is
      analyzed.
    file_path : str
      The destination path for the .txt file containing the results of the
      analysis.
    directory1 : str, optional
      The destination directory for the annotated images, comprised of
      a side-by-side view of the original image and the segmented one.
    ret : bool, optional
      If the results need to be used directly as a list, the option is there.
    """
    t0 = time.time()
    glob_path = directory0+'/*'
    paths = glob.glob(glob_path)
    results = []
    if len(paths)<n:
      n = len(paths)
    # If no second file path is provided, proceed as normal
    if directory1=="":
      f = open(file_path, "w")
      i=0
      while i<n:
        res = self.analyze(cv2.imread(paths[i]))
        f.write(str(res)+"\n")
        if ret:
          results.append(res)
        print(str(round(100*(i/n), 2))+"%")
        i+=1
      print("Running these analyses took:", time.time()-t0, "seconds.")
      f.close()
      if ret:
        return results

    # If a second file path is provided, the annotated images must be stored
    else:
      f = open(file_path, "w")
      i=0
      while i<n:
        image_file = paths[i].split(directory0+"/")[-1]
        name = "Comparison_"+str(i)+"_"+image_file.split(".")[0]+".jpg"
        new_img_path = directory1+"/"+name
        res, comparison = self.analyze(cv2.imread(paths[i]), True, [])
        f.write(str(res)+"\n")
        cv2.imwrite(new_img_path, comparison)
        if ret:
          results.append(res)
        print(str(round(100*(i/n), 2))+"%")
        i+=1
      print("Running these analyses took:", time.time()-t0, "seconds.")
      f.close()
      if ret:
        return results

  def analyze_multiple_time(self, directory0:str, n:int):
    """
    Method used to measure the time taken to analyze images at different sizes.
    This specific method only works on one repository, but was called multiple
    times for multiple different directories in order to compare the image
    sizes. The results are presented in the pdf document:
    "Image Analysis Internship - database inspection".

    Parameters:
    -----------
    directory0 : str
      The path of the directory containing the collection of images
      to be analyzed.
    n : int
      The number of images to be analyzed from the directory. If it
      surpasses the size of the directory, then the entire directory is
      analyzed.
    """
    t0 = time.time()
    glob_path = directory0+'/*'
    paths = glob.glob(glob_path)
    if len(paths)<n:
      n = len(paths)
    i=0
    while i<n:
      res = self.analyze(cv2.imread(paths[i]))
      i+=1
    # 50px is one of the sizes studied, the specific format of the names for
    # the image directories dictates the necessity of this step.
    if directory0[-5:-2] == "s50":
      size = "50"
    else:
      size = directory0[-5:-2]
    # The results were copied directly from the console, in order to save time
    # on writing another function to run the tests and properly store them.
    print(size, round(time.time()-t0, 2))

class Results_Presenter:
  """
  A class for consolidating current and future methods of presenting,
  organizing, or analyzing the results of the Image_Analyzer class.
  """

  def analysis_multiple2list(path: str):
    """
    Transforms the .txt files created by the analyze_multiple() method in
    Image_Analyzer() back into a list of lists (one sub-list for each
    image).

    Parameters:
    -----------
    path : str
      The path to the text file to be transformed back into a list.
    """
    analysis_list = []
    with open(path, "r") as f:
        for line in f:
            img_list = []
            elts = re.split('{|}', line)
            for elt in elts:
                if elt not in ['[', ']\n', ', ']:
                    elt = '{'+elt+'}'
                    elt = elt.replace("'", "\"")
                    dico = json.loads(elt)
                    img_list.append(dico)
            analysis_list.append(img_list)
    return analysis_list

  def list2table(results:list):
    """
    Presents the list of objects detected in an image in a pandas table.

    Parameters:
    -----------
    results : list
      The list returned by the analyze() method in Image_Analyzer().
    """
    results_copy = copy.deepcopy(results)
    index_labels = []
    for res in results_copy:
      index_labels.append(res['category'])
      res.pop('category')
    df = pd.DataFrame(results_copy)
    df.index = index_labels
    return df

  def list2sheet(lst:list, path:str):
    """
    The method used to transform the analysis data into the .txt file used for
    closer inspection of 1000 images.

    Parameters:
    -----------
    lst : list
      The list returned by analyze_multiple() in Image_Analyzer().
    path : str
      The file path for the future .txt file to be created.
    """
    with open(path, "w") as f:
      i = 0
      for img_list in lst:
        if i>3000:
          num = i
          if len(list(str(i)))<4:
            num = (4-len(list(str(i))))*"0"+str(i)
          img_name = "Comparison "+str(num)+":\n"
          f.write(img_name)
          for obj in img_list:
            if "confidence" in obj.keys():
              entry_name = obj["category"]
              # The longest name is 14 characters long
              if len(obj["category"])<14:
                entry_name += (14-len(obj["category"]))*" "
              val = str(round(obj["confidence"], 2))
              if len(val) < 4:
                val += (4-len(val))*"0"
              if round(obj["confidence"], 2)==1.0:
                val = "1.00"
              entry_name += " "+val+" "+"\n"
              f.write(entry_name)
          f.write("\n")
        i+=1

  def list2sheet2(lst: list, path: str):
    """
    The method used to transform the analysis data from each set of 90 images
    (at 5 different sizes) into .txt files to be manually commented. It was
    run multiple times, once for each size (each one having its own list).

    Parameters:
    -----------
    lst : list
      The list returned by the analyze_multiple() method in Image_Analyzer().
    path : str
      The file path for the future .txt file to be created.
    """
    with open(path, "w") as f:
        i = 0
        for img_list in lst:
            # Keeping a consistent format is key to the work
            if len(list(str(i))) < 4:
                num = (4-len(list(str(i))))*"0"+str(i)
                # Now the image name line will always be the same length
                img_name = "Comparison "+str(num)+":\n"
                f.write(img_name)
                # Here the image's content is studied
                for obj in img_list:
                    # If it is not a thing, it is not interesting here
                    if "confidence" in obj.keys():
                        entry_name = obj["category"]
                        if len(obj["category"]) < 14:
                            entry_name += (14-len(obj["category"]))*" "
                        val = str(round(obj["confidence"], 2))
                        if len(val) < 4:
                            val += (4-len(val))*"0"
                        if round(obj["confidence"], 2) == 1.0:
                            val = "1.00"
                        entry_name += " "+val+" "+"\n"
                        f.write(entry_name)
                    else:
                        entry_name = obj["category"]
                        if len(obj["category"]) < 14:
                            entry_name += (14 - len(obj["category"])) * " "
                            val = str(round(obj["area"], 2))
                            if len(val) < 4:
                                val += (4-len(val))*"0"
                        entry_name += " "+val.replace(".", ",")+" "+"\n"
                        f.write(entry_name)
                f.write("\n")
            i += 1

  def word_vectors(results:list, min_conf:float, min_area:float):
    """
    Filters the results of a COCO image analysis by confidence score for
    'things' and area for 'stuff'. It then generates two word vectors,
    their keys are the names of the classes found in the image.
    The things_vector has the number of instances of each thing class for its
    values.
    For the stuff_vector, the value is 10x the number of min_area (a small unit
    of area that is fixed in parameter as a filter) that the stuff class takes
    up in the image. This means the value seen in the vector, is out of
    10x(1/min_area). For example: min_area=0.2 => 50 total possible units of
    area, the stuff_vector values summed up cannot surpass 50.

    Parameters:
    -----------
    results : list
      The list returned by the analyze() method in Image_Analyzer().
    min_conf : float
      The threshold of confidence score needed to be a valid entry.
    min_area : float
      The proportion of an image that needs to be occupied to be valid.
    """
    things_vector = {}
    stuff_vector = {}
    for obj in results:
      # Keep likely things
      if ('confidence' in obj.keys()) and (obj['confidence']>=min_conf):
        if obj['category'] in things_vector.keys():
          things_vector[obj['category']] += 1
        else:
          things_vector[obj['category']] = 1
      # Keep large-enough stuff
      elif ('confidence' not in obj.keys()) and (obj['area']>=min_area):
        stuff_vector[obj['category']] = int(round(10*obj['area']/min_area, 0))
    return things_vector, stuff_vector

  def sheet2statistics(path: str):
    """
    Turns the comments collected from closer inspection of ~20% of the
    images data into graphs for selecting the best threshold for the
    model. This threshold refers to the confidence threshold for
    identification of 'things' (cf. COCO). The model (by default) gives
    scores ranging from 50% to 100%, with far more errors closer to
    50% than 100%. The aim of these graphs is for human operators to
    decide what percentage is acceptable for placing trust in the model's
    predictions. The mentions in the graphs are:
    True/T:
      When a 'thing' (cf. COCO) is correctly identified in normal
      circumstances, as opposed to art (cf. definition in
      "Image Analysis Internship - database inspection")
    False/F:
      When a thing is incorrectly identified in normal circumstances.
    True*/T*:
      When a thing is correctly identified in either normal circumstances
      or in an artistic representation. The latter of which the model was
      not designed for.
    False*/F*:
      When a thing is incorrectly identified in either normal or artistic
      representation.

    Parameters:
    -----------
    path : str
      The path for the document containing the manually commented information.
      The specific one used in the document is "Image_Comparisons.txt".
    """

    # List of all confidence scores labeled as true: True, normal.
    list_T = []
    # List of all confidence scores labeled as false: False, normal.
    list_F = []
    # List of all confidence scores labeled as correct: True, art.
    list_C = []
    # List of all confidence scores labeled as incorrect: False, art.
    list_I = []

    # Bellow is a collection of dictionaries for T, F, C, and I.
    # They compile the frequencies of confidence scores from 50% to 100%.
    dict_T = {}
    dict_F = {}
    dict_C = {}
    dict_I = {}

    # Ratios of occurences of T divided by occurences of F at a given
    # confidence score.
    dict_R = {}
    # Ratios of occurences of T* divided by occurences of F* at a given
    # confidence score.
    dict_R2 = {}
    # For a given confidence score, the proportion of T that has an equal
    # or higher confidence.
    dict_PT = {}
    # For a given confidence score, the proportion of F that has an equal
    # or higher confidence.
    dict_PF = {}
    # For a given confidence score, the proportion of T or F that has an
    # equal or higher confidence.
    dict_PTot = {}
    # For a given confidence score, the proportion of T* that has an equal
    # or higher confidence.
    dict_PT2 = {}
    # For a given confidence score, the proportion of F* that has an equal
    # or higher confidence.
    dict_PF2 = {}
    # For a given confidence score, the proportion of T* or F* that has an
    # equal or higher confidence.
    dict_PTot2 = {}

    # Initialize the keys, ranging from 50% to 100% confidence score,
    # formatted like the path doc.
    for i in range(51):
        dict_T[round(0.5 + i * 0.01, 2)] = 0
        dict_F[round(0.5 + i * 0.01, 2)] = 0
        dict_C[round(0.5 + i * 0.01, 2)] = 0
        dict_I[round(0.5 + i * 0.01, 2)] = 0
        dict_R[round(0.5 + i * 0.01, 2)] = 0
        dict_R2[round(0.5 + i * 0.01, 2)] = 0
    with open(path, "r") as f:
        # The path document was designed to make automatic reading of the
        # desired information easy. This manifests as intentional line lengths
        # to separate data or titles.
        for line in f:
            # Only lines containing the comments are longer than 17 characters.
            if len(line) > 17:
                # Only the last part of the line contains the desired
                # information: score and comment (T, F, etc.)
                res = line[15:-1]
                # To save typing time, if a 'thing' is T, no comment was left
                if res[-1] == " ":
                    dict_T[float(res[:-1])] += 1
                    list_T.append(float(res[:-1]))
                elif res[-1] == "F":
                    dict_F[float(res[:-2])] += 1
                    list_F.append(float(res[:-2]))
                elif res[-1] == "C":
                    dict_C[float(res[:-2])] += 1
                    list_C.append(float(res[:-2]))
                elif res[-1] == "I":
                    dict_I[float(res[:-2])] += 1
                    list_I.append(float(res[:-2]))

    for i in dict_R.keys():
        dict_PT[i] = sum(dict_T[j] for j in list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(list_T)
        dict_PF[i] = sum(dict_F[j] for j in list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(list_F)
        dict_PTot[i] = sum(dict_T[j] + dict_F[j] for j in list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(
            list_T + list_F)
        dict_PT2[i] = sum(dict_T[j] + dict_C[j] for j in list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(
            list_T + list_C)
        dict_PF2[i] = sum(dict_F[j] + dict_I[j] for j in list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(
            list_F + list_I)
        dict_PTot2[i] = sum(dict_T[j] + dict_C[j] + dict_F[j] + dict_I[j] for j in
                            list(dict_R.keys())[list(dict_R.keys()).index(i):]) / len(list_T + list_C + list_F + list_I)
        if dict_F[i] != 0:
            dict_R[i] = round(dict_T[i] / dict_F[i], 2)
        if dict_F[i] + dict_I[i] != 0:
            dict_R2[i] = round((dict_T[i] + dict_C[i]) / (dict_F[i] + dict_I[i]), 2)
        else:
            dict_R[i] = 0
            dict_R2[i] = 0

    plt.subplot(2, 3, 1)
    plt.hist(list_T, bins=51, color="blue")
    plt.hist(list_F, bins=51, color="red")
    plt.title("Histograms of True and False")
    plt.xlabel("Confidence score")
    plt.ylabel("Frequency")
    plt.legend(["True", "False"])
    plt.xlim([0.5, 1])

    plt.subplot(2, 3, 2)
    # The line graph was chosen for legibility
    plt.plot([x for x in list(dict_R.keys()) if dict_R[x] != 0], [y for y in list(dict_R.values()) if y != 0])
    # This horizontal line represents an arbitrary distinction: when the desired category
    # is 10x more likely than the undesired one.
    plt.plot(list(dict_R.keys()), np.linspace(10, 10, 51), color="red")
    # The scatter plot serves to better isolate the data points
    plt.scatter([x for x in list(dict_R.keys()) if dict_R[x] != 0], [y for y in list(dict_R.values()) if y != 0], s=3)
    plt.title("Ratio of True/False")
    plt.xlabel("Confidence score")
    plt.legend(["T/F", "T/F=10"])
    plt.xlim([0.5, 1])
    plt.ylim(bottom=0)

    plt.subplot(2, 3, 3)
    plt.plot(list(dict_R.keys()), list(dict_PT.values()), color="blue")
    plt.plot(list(dict_R.keys()), list(dict_PF.values()), color="red")
    plt.plot(list(dict_R.keys()), list(dict_PTot.values()), color="purple")
    # The following two vertical lines represent two possible thresholds.
    plt.plot(np.linspace(0.8, 0.8, 51), np.linspace(0, 1, 51), color="black")
    plt.plot(np.linspace(0.9, 0.9, 51), np.linspace(0, 1, 51), color="black")
    plt.title("Remaining data above threshold")
    plt.xlabel("Confidence score")
    plt.legend(["T", "F", "T+F"])
    plt.xlim([0.5, 1])
    plt.ylim([0, 1])

    # The following three graphs function the same as the previous three,
    # but include the artistic data as well.
    plt.subplot(2, 3, 4)
    plt.hist(list_T + list_C, bins=51, color="green")
    plt.hist(list_F + list_I, bins=51, color="orange")
    plt.title("Histogram of True* vs False*")
    plt.xlabel("Confidence score")
    plt.ylabel("Frequency")
    plt.legend(["True*", "False*"])
    plt.xlim([0.5, 1])

    plt.subplot(2, 3, 5)
    plt.plot([x for x in list(dict_R2.keys()) if dict_R2[x] != 0], [y for y in list(dict_R2.values()) if y != 0])
    plt.plot(list(dict_R.keys()), np.linspace(10, 10, 51), color="red")
    plt.scatter([x for x in list(dict_R2.keys()) if dict_R2[x] != 0], [y for y in list(dict_R2.values()) if y != 0],
                s=3)
    plt.title("Ratio of True*/False*")
    plt.xlabel("Confidence score")
    plt.legend(["T*/F*", "T*/F*=10"])
    plt.xlim([0.5, 1])
    plt.ylim(bottom=0)

    plt.subplot(2, 3, 6)
    plt.plot(list(dict_R.keys()), list(dict_PT.values()), color="green")
    plt.plot(list(dict_R.keys()), list(dict_PF.values()), color="orange")
    plt.plot(list(dict_R.keys()), list(dict_PTot.values()), color="brown")
    # The following two vertical lines represent two possible thresholds.
    plt.plot(np.linspace(0.8, 0.8, 51), np.linspace(0, 1, 51), color="black")
    plt.plot(np.linspace(0.9, 0.9, 51), np.linspace(0, 1, 51), color="black")
    plt.title("Remaining data* above threshold")
    plt.xlabel("Confidence score")
    plt.legend(["T*", "F*", "T*+F*"])
    plt.xlim([0.5, 1])
    plt.ylim([0, 1])

    plt.tight_layout()
    plt.show()

  def influence_resolution(path_sheets: str, path_time: str):
    """
    Turns the comments collected from closer inspection of 90 different images
    each at 5 different sizes (for a total of 450 images) into graphs for
    selecting the best size for future images.

    Parameters:
    -----------
    path_sheets : str
      The path for the directory containing the different sheets with comments
      on all 5 sizes.
    path_time : str
      The path for the document containing the time data produced by the
      Image_Analyzer method analyze_multiple_time().
    """
    list_docs = glob.glob(path_sheets + "/*")
    across_T = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_F = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_T_list = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_F_list = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_PT = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_R = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_ratio = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    across_U = {'50px': 0, '100px': 0, '200px': 0, '400px': 0, '800px': 0}
    time_dict = {'50px': [], '100px': [], '200px': [], '400px': [], '800px': []}
    colors = {'50px': 'red', '100px': 'orange', '200px': 'green', '400px': 'blue', '800px': 'purple'}
    with open(path_time, "r") as f:
        for line in f:
            val = float(line[-5:])
            if line[-5] == " ":
                val = float(line[-4:])
            if line[:2] == "50":
                time_dict["50px"].append(val)
            else:
                time_dict[line[:3]+"px"].append(val)
    for doc in list_docs:
        size = doc[-8:-5]+"px"
        if doc[-7:-5] == "50":
            size = "50px"
        # List of all confidence scores labeled as T, true.
        list_T = []
        # List of all confidence scores labeled as F, false.
        list_F = []
        # List the unidentified area in an image.
        list_U = []

        dict_T = {}
        dict_F = {}
        dict_PT = {}
        dict_R = {}
        # Initialize the keys.
        for i in range(51):
            dict_T[round(0.5 + i * 0.01, 2)] = 0
            dict_F[round(0.5 + i * 0.01, 2)] = 0
            dict_PT[round(0.5 + i * 0.01, 2)] = 0
            dict_R[round(0.5 + i * 0.01, 2)] = 0
        with open(doc, "r") as f:
            for line in f:
                # Only lines containing the comments are longer than 17 characters.
                if len(line) > 17:
                    # Only the last part of the line contains the desired information: score and comment (T, F, etc.)
                    res = line[15:-1]
                    # To save typing time, if a 'thing' is T, no comment was left
                    if res[-1] == " " and res[1] == ".":
                        dict_T[float(res[:-1])] += 1
                        across_T[size] += 1
                        list_T.append(float(res[:-1]))
                    elif res[-1] == "F" and res[1] == ".":
                        dict_F[float(res[:-2])] += 1
                        across_F[size] += 1
                        list_F.append(float(res[:-2]))
                    elif line[:4] == "unid":
                        val = float(res[:4].replace(",", "."))
                        list_U.append(val)
        across_U[size] = np.mean(list_U)
        across_T_list[size] = list_T
        across_F_list[size] = list_F
        for i in dict_T.keys():
            dict_PT[i] = sum(dict_T[j] for j in list(dict_T.keys())[list(dict_T.keys()).index(i):]) / len(list_T)
            if dict_F[i] != 0:
                dict_R[i] = round(dict_T[i] / dict_F[i], 2)
            else:
                dict_R[i] = 0
        across_PT[size] = dict_PT
        across_R[size] = dict_R
    for key in across_T.keys():
        across_ratio[key] = across_T[key] / across_F[key]
        across_T[key] = round(across_T[key]/across_T['800px'], 2)

    # Bellow, several print lines are commented, they were used to obtain the
    # values manually added to the bar graphs in the presentation document.
    # The spacing methods for matplotlib did not allow for easy display.

    plt.subplot(2, 3, 1)
    #print("1", across_T)
    plt.bar(across_T.keys(), across_T.values())
    plt.title("'Thing' predictions by image size")
    plt.xlabel("Image size")
    plt.ylabel("Number of T predictions, normalized*")

    plt.subplot(2, 3, 2)
    #print("2", across_ratio)
    plt.bar(across_ratio.keys(), across_ratio.values(), color="orange")
    plt.title("Ratios of T totals over F totals by image size")
    plt.xlabel("Image size")
    plt.ylabel("Total T predictions / total F predictions")

    plt.subplot(2, 3, 3)
    #print("3", across_U)
    plt.bar(across_U.keys(), across_U.values(), color="red")
    plt.title("Unidentified area by image size")
    plt.xlabel("Image size")
    plt.ylabel("Unidentified proportion of image")

    plt.subplot(2, 3, 4)
    for size in ["800px", "400px", "200px", "100px", "50px"]:
        plt.plot(across_PT[size].keys(), across_PT[size].values(), color=colors[size])
    plt.title("Remaining T above threshold")
    plt.xlabel("Confidence score threshold")
    plt.ylabel("Remaining T")
    plt.legend(["800px", "400px", "200px", "100px", "50px"])
    plt.xlim([0.5, 1])
    plt.ylim([0, 1.0])

    plt.subplot(2, 3, 6)
    time_avgs = {}
    for size in time_dict.keys():
        time_avgs[size] = 90/np.mean(time_dict[size])
    #print("6:", time_avgs)
    plt.bar(time_avgs.keys(), time_avgs.values(), color="green")
    plt.title("Analysis speed by image size")
    plt.xlabel("Image size")
    plt.ylabel("Images per second")

    plt.tight_layout()
    plt.show()
