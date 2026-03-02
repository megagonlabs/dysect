#from t5 import T5Probe
import argparse
import json
import random
import itertools
import ast
from pathlib import Path
import re
import os
import shutil
import spacy
from sentence_transformers import SentenceTransformer
#import vLLMlib
import pandas as pd
import numpy as np
import unicodedata
from openai import OpenAI
import os
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from scipy.special import expit, logit


spacyModel = spacy.load("en_core_web_lg")
sentenceBertModel = SentenceTransformer('all-MiniLM-L6-v2')
from datetime import datetime

class MyLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MyLogger, cls).__new__(cls)
            cls._instance.log_file_name = "logFile.log"
        return cls._instance

    def configure(self, log_file_name):
        self.log_file_name = log_file_name

    def log(self, message, level="INFO"):
        level = level.upper()
        if level not in {"INFO", "WARNING", "ERROR"}:
            raise ValueError("Invalid log level. Use 'INFO', 'WARNING', or 'ERROR'.")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        with open(self.log_file_name, "a", encoding="utf-8") as f:
            f.write(log_entry)
        #print(log_entry) # also shows the message on the screen


def splitConceptIntoSubconcepts(generalizationConcept,startOfCurrentIterationDate,ingestionIteration, knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources):
    #generalizationConcept = 'skills'
    listOfElements = getValue(generalizationConcept,'specializations', knowledgeBaseID,knowledgeBaseIDPath)
    #llmResponse = bLib.llmBasedCLuster('skills','generalizations', listOfElements.keys(), knowledgeBaseID)
    embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
    proposedClusters =  entityListClusteredByEmbeddingSimilarity(listOfElements,embeddingModel)
    sourceOfInformation = 'subconcept generator based on SentenceTransformer with all-MiniLM-L6-v2 embeddings and Kmeans cluster' 
    confidence = '0.4'
    splitConcept(generalizationConcept, proposedClusters, knowledgeBaseID,ingestionIteration,sourceOfInformation,startOfCurrentIterationDate,confidence,knowledgeBaseIDPath,fullyTrustedSources)
    

def getNumberOfConceptsAndRelations(knowledgeBaseID,knowledgeBaseIDPath):
    numberOfconceptsAndInstances = 0
    entitiesInKB = getValue('conceptsToBeLearned','specializations', knowledgeBaseID,knowledgeBaseIDPath)
    for entityToBeCounted in entitiesInKB:
        numberOfconceptsAndInstances += 1




def openai_response(model_name, messages, client, sampling_params, response_format=None):
    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "max_tokens": sampling_params["max_tokens"],
    }
    if response_format:
        request_params["response_format"] = response_format
        chat_completion = client.beta.chat.completions.parse(**request_params)
    else:
        chat_completion = client.chat.completions.create(**request_params)
    return chat_completion.choices[0].message.content


def loadParameters(propertiesFileName, knowledgeBaseID, knowledgeBasePath, lastIteration):
    if knowledgeBaseID != 'none':
        MyLogger().log('KnowledgeBaseID was provided as '+ str(knowledgeBaseID) + 'provided in the code. Loading the parameters.json file from the respective knowledgeBaseID path.', level='INFO')
        propertiesJsonFileName = knowledgeBasePath+ str(knowledgeBaseID) + "/" + str(propertiesFileName) 
        MyLogger().log('Full path: '+ str(propertiesJsonFileName), level='INFO')
        with open(propertiesJsonFileName, 'r') as propertiesJsonFile:
            properties = json.load(propertiesJsonFile)
        properties[0]['lastIteration'] = lastIteration
        with open(propertiesJsonFileName, 'w') as propertiesJsonFile:
            json.dump(properties, propertiesJsonFile, indent = 4)
        MyLogger().log('Parameters Loaded:' + str(properties), level='INFO')
        return properties
    else:
        with open(str(propertiesFileName), 'r') as propertiesJsonFile:
            properties = json.load(propertiesJsonFile)
        knowledgeBaseID = properties[0]['knowledgeBaseID']
        MyLogger().log("No hardcoded KnowledgeBaseID provided. Using the ID provided in the parameters.json file provided in the root path.", level="INFO")
        MyLogger().log('Parameters Loaded:' + str(properties), level='INFO')
        kbStorageInitialization(knowledgeBaseID)
        return properties

            
def getEntitiesFromT5(initialList,concept,numberOfExtractions,prober,iterationNumber,knowledgeBaseID,prompts):
    #entityListPairs = list(itertools.combinations(initialList, 2))
    entityListPairs = entityListPairsByWordEmbeddingSimilarityFetchingSecondElementOfThePair(initialList)#entityListPairsByRandomlyFetchingSecondElementOfThePair(initialList)
    promptTemplates = fetchPromptsFromFile(prompts)
    promptList = []
    #knowledgeBaseID = str(datetime.now().isoformat()).split('T')[0]
    return probe_T5_ModelWithGenericPrompt(entityListPairs,promptTemplates,promptList,concept,numberOfExtractions,prober,'db',iterationNumber,str(knowledgeBaseID))

def loadProbers():
    prober = 'T5Probe(model_name_or_path="t5-large")'#TODO: deprecate and delete this function

def entityListPairsBySimpleCombination(initialList):
    entityListPairs = list(itertools.combinations(initialList, 2))
    return entityListPairs

def entityListPairsByRandomlyFetchingSecondElementOfThePair(listOfElements):
    entityPairsList = []
    randomElement = ''
    for listElement in listOfElements:
        randomElement = listElement
        while randomElement == listElement:
            randomElement = random.choice(listOfElements)
        entityPairsList.append((listElement,randomElement))
    return entityPairsList

def entityListPairsByWordEmbeddingSimilarityFetchingSecondElementOfThePair(listOfElements):
    entityPairsList = []
    for firstElement in listOfElements:
        firstElementEmbedding = spacyModel(firstElement)
        higherSimilarityScore = 0
        for secondElement in listOfElements:
            if not firstElement==secondElement:
                secondElementEmbedding = spacyModel(secondElement)
                currentSimilarityScore = firstElementEmbedding.similarity(secondElementEmbedding)
                if currentSimilarityScore > higherSimilarityScore:
                    higherSimilarityScore = currentSimilarityScore
                    secondElementChosen = secondElement
        entityPairsList.append((firstElement,secondElementChosen))
    return entityPairsList

def distanceFromCentroid(kmeans):
    pass

def entityListClusteredByEmbeddingSimilarity(listOfElements,embeddingModel):
    if listOfElements != None and len(listOfElements) > 10:
        listOfClusters = []
        keysFromListOfElements = list(listOfElements.keys())
        listOfEmbeddings = generateListOfEmbeddings(keysFromListOfElements,embeddingModel)
        numberOfClusters = int(len(listOfElements)*0.2) #TODO: review the strategy for defining the number of clusters. For now we are going more agressive in the number of clusters and that will results in clusters with smaller number of instances in each cluster, which can help to extract more focused examples.
        MyLogger().log('Applying kmeans clustering using '+str(numberOfClusters)+ ' clusters in this list: '+str(keysFromListOfElements))
        kmeans = KMeans(n_clusters=numberOfClusters, random_state=0,n_init=10).fit(np.array(listOfEmbeddings))
        for clusterID in range(numberOfClusters):
            elementsIntheCluster = np.where(kmeans.labels_ == clusterID)[0].tolist()
            tempList = []
            finalList = []
            MyLogger().log('Finding the distance of each element in the cluster to its centroid.')
            averageDistanceFromTheCentroid = 0
            for elementInTheCluster in elementsIntheCluster: 
                centroidValue = kmeans.cluster_centers_[clusterID]
                distanceFromCentroid = np.linalg.norm(listOfEmbeddings[elementInTheCluster] - centroidValue)
                averageDistanceFromTheCentroid = averageDistanceFromTheCentroid + distanceFromCentroid
                tempList.append((keysFromListOfElements[elementInTheCluster],distanceFromCentroid))
            averageDistanceFromTheCentroid = float(averageDistanceFromTheCentroid) / float(len(elementsIntheCluster))
            MyLogger().log('filtering out elements that are farther than the average distance from the centroid.')
            for elementInTempList in tempList:
                if elementInTempList[1] < averageDistanceFromTheCentroid:
                    finalList.append(elementInTempList[0])
            listOfClusters.append(finalList)
        return listOfClusters
    else:
        return None

def llmBasedClusterNaming(clusterElements,generalizationConcept):
    pass
    #TODO: probe llm to provide a "name" to the concept that represents the subset of elements from the generalization (or parent) concept
    gptPrompt0 = 'You will be given a general category and a list of instances that belong to a more specific subcategory within it. Identify and output only the name of that subcategory, including the main category name as part of it. Do not provide any explanations—output the subcategory name only. \n\nCategory: '
    llamaPrompt0 = 'Task: Determine the specific subcategory that a set of instances belongs to, based on the given general category. The subcategory must include the main category name as part of its label (e.g., "sedan vehicles"). Respond with the subcategory name only. Do not include any explanations or additional text. \\Category: '
    prompt1 = str(generalizationConcept)
    prompt2 = '\n\nInstances: ' + str(clusterElements).replace('[','').replace(']','')
    gptPrompt3 = '\n\nSubcategory: '
    llamaPrompt3 = '\n\nAnswer: '
    fullLlamaPrompt = llamaPrompt0 + prompt1 + prompt2 + llamaPrompt3
    fullgptPrompt = gptPrompt0 + prompt1 + prompt2 + llamaPrompt3
    openAImodel = 'gpt-4o-mini'
    openAIresponse = probeOpenAI(fullgptPrompt,openAImodel)
    #llamaResponse = probevLLM(fullLlamaPrompt)
    MyLogger().log(openAIresponse)
    return openAIresponse, gptPrompt0, openAImodel 

def hierarchyMaintainance(generalizationConcept):
    MyLogger().log('checking for hierarchy inconsistencies')

def conceptIsInKB(concept,relationPredicate,knowledgeBaseID, knowledgeBasePath):
    valueGotten = getValue(concept,relationPredicate,knowledgeBaseID,knowledgeBasePath)
    if valueGotten != None:
        return valueGotten
    else:
        return False
def getParameterFromFile(targetParameter, knowledgeBaseID, knowledgeBaseIDPath):
    rootParametersFileName = 'parameters.json'
    parametersFileName = knowledgeBaseIDPath + str(knowledgeBaseID) + '/' + str(rootParametersFileName)
    try:
        with open(parametersFileName, "r", encoding="utf-8") as parametersFile:
            parametersData = json.load(parametersFile)
            return parametersData[0][str(targetParameter)]
    except:
        return None

def splitConcept(generalizationConcept, splittedCLusters,knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence, knowledgeBasePath,fullyTrustedSources):
    if splittedCLusters != None and len(splittedCLusters)>10:
        if currentIteration > 1:
            clusterSizeThreshold = getParameterFromFile('subconceptGenerationClustersSize', knowledgeBaseID, knowledgeBasePath)
        else:
            clusterSizeThreshold = 5
        for listOfClusteredElements in splittedCLusters:
            if listOfClusteredElements != None and len(listOfClusteredElements)>clusterSizeThreshold:
                MyLogger().log('processing the proposed clustering containing: '+str(listOfClusteredElements))
                clusterName, llmPrompt, llmModel = llmBasedClusterNaming(listOfClusteredElements,generalizationConcept)
                MyLogger().log('cluster name proposed by LLM: '+str(clusterName))
                clusterNameGeneralizations = conceptIsInKB(clusterName,'generalizations',knowledgeBaseID,knowledgeBasePath)
                #TODO: change the check above
                MyLogger().log('Original prompt: '+str(llmPrompt))
                #MyLogger().log('replacing the new line character "\n" with space in the provenance to avoid problems when serializing the json.')
                provenanceInfo = str(llmModel)+'||elements||'+ str(listOfClusteredElements) + '||prompt||'+str(llmPrompt.replace('\n','\\n'))
                MyLogger().log('prompt stored as provenance: '+str(llmPrompt))
                if clusterNameGeneralizations:
                    MyLogger().log('subconcept '+str(clusterName)+' was already present in the KB.')
                    if clusterNameGeneralizations != generalizationConcept:
                        MyLogger().log(str(generalizationConcept)+' is a new generalization and will be added to '+str(clusterName))
                    else:
                        MyLogger().log('The metadata about this extra mention to '+str(clusterName)+ 'was alredy prosent and frequency and confidence are now updated.')
                else:
                    MyLogger().log('New subconcept added: '+ str(clusterName))
                    MyLogger().log('New subconcept generalization: ' + str(generalizationConcept))                   
                confidence = confidence #TODO: the confidence should be based on the individual confidence of each cluster element
                addValueInverseAndTypes(clusterName,'generalizations',generalizationConcept,knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,provenanceInfo, knowledgeBasePath,fullyTrustedSources)
                conceptsToBeLearnedSpecializations = getValue('concepts to be learned in the current kb','specializations',knowledgeBaseID,knowledgeBasePath) 
                if clusterName in conceptsToBeLearnedSpecializations:
                    MyLogger().log('The concept '+str(clusterName)+' was already present in the KB, but it was not yet an specialization of '+str("concepts to be learned in the current KB")+'.\nJust adding the metadata about this extra mention to it.')
                else:
                    MyLogger().log('New concept added as specialization of the concept "concepts to be learned in the current kb": '+ str(clusterName))
                addValueInverseAndTypes(clusterName,'generalizations','concepts to be learned in the current kb',knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,provenanceInfo, knowledgeBasePath,fullyTrustedSources)  
                for clusteredElement in listOfClusteredElements:
                    addValueInverseAndTypes(clusteredElement,'generalizations',clusterName, knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,provenanceInfo, knowledgeBasePath,fullyTrustedSources)                
                conceptSpecializations = hierarchyMaintainance(generalizationConcept)
    else:
        MyLogger().log('empty cluster. Skipping the subconcept generation')

def clusterForEmbeddingSimilarity(listOfElements,embeddingModel):
    listOfEmbeddings = generateListOfEmbeddings(listOfElements,embeddingModel)
    entityPairsList = []
    if len(listOfElements) < 5: #TODO: need to plan on the strategy of how many clusters are needed. In this first try, I'm fixing 3 clusters and I'll select at most 100 elements from each, thus, in total for each concept every iteration will prompt using 300 seed instances
        return False
    else:
        numberOfClusters = 4 #TODO: we are starting with the creation of 4 new subconcepts everytime a concept achieves 
    kmeans = KMeans(n_clusters=numberOfClusters, random_state=0,n_init=10).fit(np.array(listOfEmbeddings))
    for cluster in range(numberOfClusters):
        elementsIntheCluster = np.where(kmeans.labels_ == cluster)[0].tolist()
        selectedElements = np.random.choice(elementsIntheCluster, 100)
        #TODO: add a probability distribution to guide the selection based on the number of time each seed has already been used
        for selectedElement in selectedElements:
            randomElement = selectedElement
            while randomElement == selectedElement:
                randomElement = np.random.choice(elementsIntheCluster, 1) 
            entityPairsList.append((selectedElement,randomElement))
    return entityPairsList

def generateListOfEmbeddings(listOfElements,sentenceBertModel):
    listOfEmbeddings = []
    for item in listOfElements:
        listOfEmbeddings.append(sentenceBertModel.encode(item))
    return listOfEmbeddings


def relationListPairsByWordEmbeddingSimilarityFetchingSecondElementOfThePair(listOfElements):
    entityPairsList = []
    for firstPair in listOfElements:
        firstPairFirstElementEmbedding = spacyModel(firstPair[0])
        firstPairSecondElementEmbedding = spacyModel(firstPair[1])
        higherSimilarityScore = 0
        for secondPair in listOfElements:
            if not firstPair==secondPair:
                secondPairFirstElementEmbedding = spacyModel(secondPair[0])
                secondPairSecondElementEmbedding = spacyModel(secondPair[1])
                currentSimilarityScore = firstPairFirstElementEmbedding.similarity(secondPairFirstElementEmbedding) + firstPairSecondElementEmbedding.similarity(secondPairSecondElementEmbedding) #for now we are considering the similarity of the pairs as the sum of the similarities of the elements of the pairs. We can adopt differentsimilarity metrics later.
                if currentSimilarityScore > higherSimilarityScore:
                    higherSimilarityScore = currentSimilarityScore
                    secondPairChosen = secondPair
        entityPairsList.append((firstPair,secondPairChosen))
    return entityPairsList

def testPrompt(prompt,modelToProbe):
    extractions = modelToProbe(prompt, topk=20, max_new_tokens=50)
    for extraction in extractions['values']:
        MyLogger().log(extraction['token'])

def kbStorageInitialization(knowledgeBaseID):
    MyLogger().log('Storing the parameters in the specific knowledgeBaseID path.', level='INFO')
    pathToCheck = "data/kbs/"+ str(knowledgeBaseID) + "/prompts/"
    if not os.path.exists(pathToCheck):
        MyLogger().log('Path: ' + str(pathToCheck) + 'does not exist. Creating it.', level='INFO')
        os.makedirs("data/kbs/"+ str(knowledgeBaseID) + "/concepts/0")
        os.makedirs("data/kbs/"+ str(knowledgeBaseID) + "/relations/0")
        os.makedirs("data/kbs/"+ str(knowledgeBaseID) + "/prompts/")
        os.makedirs("data/kbs/"+ str(knowledgeBaseID) + "/kb/")
    #as the prompts are concept (and relation) agnostic we load the 'prompt.json' file and the prompt_relations.json file
    shutil.copy2("prompts/prompt.json", "data/kbs/"+ str(knowledgeBaseID) + "/prompts/")
    shutil.copy2("prompts/prompt.json", "data/kbs/"+ str(knowledgeBaseID) + "/concepts/0")
    shutil.copy2("prompts/prompt_relation_vllm.json", "data/kbs/"+ str(knowledgeBaseID) + "/prompts/")
    shutil.copy2('parameters.json', "data/kbs/"+ str(knowledgeBaseID) + "/")
    MyLogger().log('Path created.', level='INFO')

    #as the seeds are concetp- and relation-dependent, the user needs to the seeds files or allow the LLM to create the initial seeds.
    #print('Now, you need to copy Seed concept files to data/kbs/'+ str(knowledgeBaseID) + '/concepts/0')
    #print('and the Seed relation files to data/kbs/'+ str(knowledgeBaseID) + '/relations/0')
    #print('\n\n Would you like to automatically create the seed files using the LLM just loaded? (y/n)')
    #if input()=='y':
    #    #TODO: create seed files using the LLM
    #    print('Seed files being created based on the LLM')    


def calibrationAfterIterationConcluded(kbElement,fileNameWithPath,knowledgeBaseID,iteration,currentDate):
    pass

#def listOfAllSpecializationsAndSubSpecializations(entity):
#    pass

def traverse_down_specializations(initialNode, allSpecializationsAndSubSpecializations,knowledgebaseID,knowledgeBaseIDPath):
    targetSpecializations = getValue(initialNode,'specializations',knowledgebaseID,knowledgeBaseIDPath)
    for targetSpecialization in targetSpecializations:
        nextLevelSpecializations = getValue(targetSpecialization, 'specializations', knowledgebaseID,knowledgeBaseIDPath)
        if nextLevelSpecializations != None:
            listOfAllSpecializationsAndSubSpecializations.update(nextLevelSpecializations)
            traverse_down_specializations(targetSpecialization, allSpecializationsAndSubSpecializations, knowledgebaseID,knowledgeBaseIDPath)
        else:
            return allSpecializationsAndSubSpecializations

def addMutuallyExclusiveConcepts(conceptToInspect,dateOfCurrentIteration,iteration,knowledgeBaseID,knowledgeBaseIDPath,mutuallyExclusivityPrompt):
    #TODO:add logging
    MyLogger().log('Getting the children for t he concept '+str(conceptToInspect)+'.')
    specializationInstances = getValue(conceptToInspect,'specializations',knowledgeBaseID,knowledgeBaseIDPath)
    if specializationInstances:
        MyLogger().log('Adding mutually exclusive concepts for each one of the found children: ')
        MyLogger().log(str(specializationInstances))
        sourceOfInformation = 'gpt-4o-mini'
        MyLogger().log('Source of Information: gpt-4o-mini')
        confidence = 0.4       
        mutuallyExclusiveInitialCandidateList = specializationInstances.keys()
        mutuallyExclusiveCandidateList = []
        for mutuallyExclusiveCandidate in mutuallyExclusiveInitialCandidateList:
            if getValue(mutuallyExclusiveCandidate,'specializations',knowledgeBaseID,knowledgeBaseIDPath): #checking whether the concept is a leaf node. Only find mutually exclusive concepts for non-leaf nodes
                mutuallyExclusiveCandidateList.append(mutuallyExclusiveCandidate)
        currentPrompt = eval(mutuallyExclusivityPrompt)
        MyLogger().log(str(mutuallyExclusivityPrompt))
        mutualExclusiveFinalList = eval(probeOpenAI(currentPrompt,'gpt-4o-mini'))
        MyLogger().log('This is the list of mutual exclusive elements: '+ str(mutualExclusiveFinalList))
        pairPromptTemplate = getMutuallyExclusivePairPrompt()
        for mutuallyExclusiveElement in mutualExclusiveFinalList:
            if len(mutuallyExclusiveElement) == 2: #if the length is not 2, it means there is an error coming from the LLM, so just ignore this instance
                doubleCheck = eval(doubleCheckMutualExclusivePairs(mutuallyExclusiveElement[0],conceptToInspect, mutuallyExclusiveElement[1], conceptToInspect, pairPromptTemplate))
                MyLogger().log('')
                if not doubleCheck:
                    doubleCheck = eval(doubleCheckMutualExclusivePairs(mutuallyExclusiveElement[1],conceptToInspect, mutuallyExclusiveElement[0], conceptToInspect, pairPromptTemplate))
                    if not doubleCheck:
                        relationPredicate = str(conceptToInspect)+'_is mutually exclusive with_'+str(conceptToInspect)
                        provenanceInfo = 'this is the original mutually exclusiveness relationship came from the original triplet||'+ str(conceptToInspect)+'||specializations||'+ str(mutuallyExclusiveElement) +'||prompt template||'+str(mutuallyExclusivityPrompt)+'||prompt||'+str(currentPrompt.replace('\n','\\n'))
                        loggerMessage = 'adding the value, source, iteration, date and confidence of the mutually exclusiveness relationship. '+ str(provenanceInfo)
                        addValueInverseAndTypes(str(mutuallyExclusiveElement[0]),str(relationPredicate), str(str(mutuallyExclusiveElement[1])),knowledgeBaseID,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,provenanceInfo,knowledgeBaseIDPath,[])
                        #addValueFullPath(str(mutuallyExclusiveElement[0]),str(relationPredicate), str(str(mutuallyExclusiveElement[1])),loggerMessage,provenanceInfo,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,knowledgeBaseID,knowledgeBaseIDPath,[])
                        #provenanceInfo = 'this is the inverse mutually exclusiveness relationship came from the original triplet||'+ str(conceptToInspect)+'||specializations||'+ str(mutuallyExclusiveElement) +'||prompt template||'+str(mutuallyExclusivityPrompt)+'||prompt||'+str(currentPrompt)
                        #addValueFullPath(str(mutuallyExclusiveElement[1]),str(relationPredicate), str(mutuallyExclusiveElement[0]),loggerMessage,provenanceInfo,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,knowledgeBaseID,knowledgeBaseIDPath,[])                
                    else:
                        promptTemplate = ''
                        addImplicationRule(mutuallyExclusiveElement[1],conceptToInspect, mutuallyExclusiveElement[0], conceptToInspect, promptTemplate)
                else:
                    promptTemplate = ''
                    addImplicationRule(mutuallyExclusiveElement[0],conceptToInspect, mutuallyExclusiveElement[1], conceptToInspect, promptTemplate)
    else:
        MyLogger().log('The concept '+str(conceptToInspect)+' has no children, so we will no add mutual exclusive elements.')


def addImplicationRule(entity1,entity1Type, entity2, entity2Type, promptTemplate):
    pass

def doubleCheckMutualExclusivePairs(entity1,entity1Type, entity2, entity2Type, promptTemplate):
    promptInstance = eval(promptTemplate)
    finalOutput = probeOpenAI(promptInstance)
    return finalOutput

def getMutuallyExclusivePairPrompt(): #TODO:this is hardcoded and must be replaced
    with open('prompts/prompt_mutualy-exclusiveness.json','r') as mutuallyExclusivePromptFile:
        fullContent = json.load(mutuallyExclusivePromptFile)
    return fullContent[0]['pairPrompt1']


def addMutuallyExclusiveConcepts_Deprecated(conceptToInspect,dateOfCurrentIteration,iteration,knowledgeBaseID,knowledgeBaseIDPath,mutuallyExclusivityPrompt):
    #TODO:add logging
    MyLogger().log('Adding mutually exclusive concepts for each one of the concepts in the follwoing list.')
    generalizationInstances = getValue(conceptToInspect,'generalizations',knowledgeBaseID,knowledgeBaseIDPath)
    MyLogger().log(str(generalizationInstances))
    sourceOfInformation = 'gpt-4o-mini'
    MyLogger().log('Source of Information: gpt-4o-mini')
    confidence = 0.1       
    for currentGeneralization in generalizationInstances:
        if currentGeneralization != 'concepts to be learned in the current kb':
            #if currentGeneralization.lower() != 'everything':
            mutuallyExclusiveCandidateFullList = list(getValue(currentGeneralization,'specializations',knowledgeBaseID,knowledgeBaseIDPath).keys())
            mutuallyExclusiveCandidateList = mutuallyExclusiveCandidateFullList
            if len(mutuallyExclusiveCandidateFullList)>10:
                mutuallyExclusiveCandidateList = random.sample(mutuallyExclusiveCandidateFullList, k=10)
            currentPrompt = eval(mutuallyExclusivityPrompt)
            MyLogger().log(str(mutuallyExclusivityPrompt))
            mutualExclusiveFinalList = eval(probeOpenAI(currentPrompt,'gpt-4o-mini'))
            MyLogger().log('This is the list of mutual exclusive elements: '+ str(mutualExclusiveFinalList))
            for mutualExclusiveElement in mutualExclusiveFinalList:
                relationPredicate = str(currentGeneralization)+'_is mutually exclusive with_'+str(currentGeneralization)
                provenanceInfo = 'this is the original triplet||'+ str(conceptToInspect)+'||'+ str(relationPredicate) + '||'+ str(mutualExclusiveElement) +'||prompt template||'+str(mutuallyExclusivityPrompt)+'||prompt||'+str(currentPrompt)
                loggerMessage = 'adding the value, source, iteration, date and confidence of the generalization for the original triplet: '+ str(provenanceInfo)
                addValueFullPath(str(conceptToInspect),str(relationPredicate), str(mutualExclusiveElement),loggerMessage,provenanceInfo,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,knowledgeBaseID,knowledgeBaseIDPath,[])
                provenanceInfo = 'this is the inverse triplet||'+ str(mutualExclusiveElement)+'||'+ str(relationPredicate) + '||'+ str(conceptToInspect) +'||prompt template||'+str(mutuallyExclusivityPrompt)+'||prompt||'+str(currentPrompt)
                addValueFullPath(str(mutualExclusiveElement),str(relationPredicate), str(conceptToInspect),loggerMessage,provenanceInfo,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,knowledgeBaseID,knowledgeBaseIDPath,[])                


def addAndCheckConceptProperties():
    pass

def knowledgeIntegrator(knowledgeBaseID,iteration, rootConcept, frequencyThreshold, promotionCriteria, dateOfCurrentIteration,knowledgeBaseIDPath,fullyTrustedSources,mutuallyExclusivityPrompt):
    pass #TODO: go through all the concepts and relations to be learned and inspect their specializations
    conceptsToInspect = getValue('concepts to be learned in the current kb','specializations',knowledgeBaseID,knowledgeBaseIDPath)
    if conceptsToInspect != None:
        MyLogger().log('The concepts to be learned that will be used to guide the KNowledge Integrator are: '+str(conceptsToInspect.keys()))
        for conceptToInspect in conceptsToInspect:
            """
            listOfElements = getValue(conceptToInspect,'specializations', knowledgeBaseID,knowledgeBaseIDPath)
            #llmResponse = bLib.llmBasedCLuster('skills','generalizations', listOfElements.keys(), knowledgeBaseID)
            embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')
            proposedClusters =  entityListClusteredByEmbeddingSimilarity(listOfElements,embeddingModel)
            sourceOfInformation = 'subconcept generator v0.9'
            confidence = 0.5 #getValue()
            promptTemplate = ''
            promptInstance = ''
            splitConcept(conceptToInspect, proposedClusters, knowledgeBaseID,iteration,sourceOfInformation,dateOfCurrentIteration,confidence,knowledgeBaseIDPath)
            """
            splitConceptIntoSubconcepts(conceptToInspect,dateOfCurrentIteration,iteration,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
            MyLogger().log('checking for mutualy exclusive concepts')
            addMutuallyExclusiveConcepts(conceptToInspect,dateOfCurrentIteration,iteration,knowledgeBaseID,knowledgeBaseIDPath,mutuallyExclusivityPrompt)
            MyLogger().log('checking description')
            addAndCheckDescriptions()
            MyLogger().log('checking for relation properties')
            addAndCheckConceptProperties()

    else:
        MyLogger().log('No concepts to be processed.')


def addAndCheckDescriptions():
    pass

def knowledgeIntegrator_deprecated(knowledgeBaseID,iteration, path, frequencyThreshold, promotionCriteria):
    listOfFiles = []
    currentDate = str(datetime.now().isoformat()).split('T')[0]
    #go over all items in the KB. The first level dir is the root level
    for firstLevelDirItem in os.listdir(path):
        if len(firstLevelDirItem.split('.json')) == 1 and os.path.isdir(path+'/'+firstLevelDirItem+'/'):
            #build the path to get to the second level dir
            firstLevelPath = path + firstLevelDirItem+'/'

            #go over all items in the second level dir
            for secondLevelDirItem in os.listdir(firstLevelPath):

                if len(secondLevelDirItem.split('.json')) == 1 and os.path.isdir(firstLevelPath+secondLevelDirItem+'/'):
                    #build the path to get to the second level dir
                    secondLevelPath = firstLevelPath +secondLevelDirItem+'/'

                    #go over all items in the third level dir
                    for thirdLevelDirItem in os.listdir(secondLevelPath):

                        if len(thirdLevelDirItem.split('.json')) == 1 and os.path.isdir(secondLevelPath+thirdLevelDirItem+'/'):
                            #build the path to get to the third level dir
                            thirdLevelPath = secondLevelPath +thirdLevelDirItem+'/'

                            #go over all items in the fourth level dir
                            for fourthLevelDirItem in os.listdir(thirdLevelPath):

                                if len(fourthLevelDirItem.split('.json')) == 1 and os.path.isdir(thirdLevelPath+fourthLevelDirItem+'/'):
                                    #build the path to get to the fourth level dir
                                    fourthLevelPath = thirdLevelPath+fourthLevelDirItem+'/'
                                
                                    #go over all items in the fith level dir
                                    for fithLevelDirItem in os.listdir(fourthLevelPath):

                                        if len(fithLevelDirItem.split('.json')) == 1 and os.path.isdir(fourthLevelPath+fithLevelDirItem+'/'):
                                            #build the path to get to the fith level dir
                                            fithLevelPath = fourthLevelPath+fithLevelDirItem+'/'

                                            for sixthLevelDirItem in os.listdir(fithLevelPath):
                                                if len(sixthLevelDirItem.split('.json')) == 2:
                                                    #build the path (including the file name) to write the file in the fith level dir
                                                    with open(fithLevelPath+sixthLevelDirItem, 'r') as jsonFile:
                                                        kbElement= json.load(jsonFile)
                                                        checkPromotionToBelief_frequencyBased(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration, promotionCriteria)
                                                        splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                                        calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                                else:
                                                    MyLogger().log('ERROR. File not found.')
                                        else:
                                            if len(fithLevelDirItem.split('.json')) == 2:
                                                #build the path (including the file name) to write the file in the third level dir
                                                with open(fourthLevelPath+fithLevelDirItem, 'r') as jsonFile:
                                                    kbElement= json.load(jsonFile)
                                                    checkPromotionToBelief_frequencyBased(kbElement,fithLevelDirItem.split('.json')[0],knowledgeBaseID,iteration, promotionCriteria)
                                                    splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                                    calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                else:
                                    if len(fourthLevelDirItem.split('.json')) == 2:
                                        #build the path (including the file name) to write the file in the third level dir
                                        with open(thirdLevelPath+fourthLevelDirItem, 'r') as jsonFile:
                                            kbElement= json.load(jsonFile)
                                            checkPromotionToBelief_frequencyBased(kbElement,fourthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration, promotionCriteria)
                                            splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                            calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                        else:
                            if len(thirdLevelDirItem.split('.json')) == 2:
                                #build the path (including the file name) to write the file in the third level dir
                                with open(secondLevelPath+thirdLevelDirItem, 'r') as jsonFile:
                                    kbElement= json.load(jsonFile)
                                    checkPromotionToBelief_frequencyBased(kbElement,thirdLevelDirItem.split('.json')[0],knowledgeBaseID,iteration, promotionCriteria)
                                    splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                                    calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                else:
                    if len(secondLevelDirItem.split('.json')) == 2:
                        #build the path (including the file name) to write the file in the second level dir
                        with open(firstLevelPath+secondLevelDirItem, 'r') as jsonFile:
                            kbElement = json.load(jsonFile)
                            checkPromotionToBelief_frequencyBased(kbElement,secondLevelDirItem.split('.json')[0],knowledgeBaseID,iteration, promotionCriteria)
                            splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                            calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
        else:
            if len(firstLevelDirItem.split('.json')) == 2:
                #build the path (including the file name) to write the file in the first level dir
                with open(firstLevelDirItem, 'r') as jsonFile:
                    kbElement = json.load(jsonFile)
                    checkPromotionToBelief_frequencyBased(kbElement,firstLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,promotionCriteria)
                    splitAndMergeConceptsAndRelations(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)
                    calibrationAfterIterationConcluded(kbElement,sixthLevelDirItem.split('.json')[0],knowledgeBaseID,iteration,currentDate)


def llmBasedCLuster(entity,relationName, listOfCurrentValues, knowledgeBaseID):
    """
    apply a clustering approach to the instances of a concept, or to the pairs of instances of a relation
    """
    clusteringPrompt = 'split the following list of elements into two groups based on their similarity. All elements in one group should be similar to each other and the elements of different groups should be dissimilar.\n\nList of elements: '+ str(listOfCurrentValues) + '\n\nPlease, format your answer as a python list with 2 sublists (each sublist contains elements of each cluster). Do not add any comment, output just the python list starting with [ and ending with ]'
    #response = probeOpenAI(clusteringPrompt)
    response = probevLLM(clusteringPrompt)
    MyLogger().log(response)
    return response

def call_gpt5_mini(prompt: str) -> str:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.responses.create(
        model="gpt-5-mini",
        input=prompt,
        max_output_tokens=512,   # controls the length of the completion
    )

    # The assistant content is inside response.output[0].content[0].text
    print('openAI Response: '+ str(response.output[0].content[0].text))
    input()

    return response.output[0].content[0].text


def probeOpenAI2(prompt, model_name="gpt-5-mini"):
    api_key=os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    if not api_key:
        MyLogger.log('The OPENAI_API_KEY environment variable is NOT set.')
    temperature = 0.0
    top_p = 1.0
    max_output_tokens = 16384

    try:
        response = openai_response(
            model_name,
            [{"role": "user", "content": prompt}],
            client,
            {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_output_tokens,
            }
        )
    except Exception as e:
        error_message = str(e)
        MyLogger().log(error_message)
        response = None

    MyLogger().log(f"{model_name} returned:\n{response}")
    return response

def probeOpenAI(prompt, model_name="gpt-4o-mini"):
    api_key=os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)
    if not api_key:
        MyLogger.log('The OPENAI_API_KEY environment variable is NOT set.')
    temperature = .0
    top_p = 1.0
    max_tokens = 16384

    #prompt = "Please, help me understanding what are the highly probable skills a job candidate has, if I know she/he has experience with python."
    try:
        response = openai_response(
            model_name, 
            [{"role": "assistant", "content": prompt}], 
            client, 
            {"temperature": temperature, 
            "top_p": top_p, 
            "max_tokens": max_tokens}
            )
    except Exception as e:
        # 'e' now holds the exception object
        error_message = str(e)  # Get the string representation of the error
        MyLogger().log(error_message)
        response = None
    MyLogger().log(str(model_name) +' returned: \n'+str(response))
    return response

def splitAndMergeConceptsAndRelations(kbElement,knowledgeBaseID,iteration,currentDate, maxNumberOfValues,knowledgeBaseIDPath,fullyTrustedSources):
    #go through all relations (keys) in the json file
    for relationName in kbElement:
        MyLogger().log('checkPromotionToBelief_frequencyBased: Checking Relation:'+str(relationName))
        listOfCurrentValues = getValue(kbElement, relationName, knowledgeBaseID,knowledgeBaseIDPath)
        if len(listOfCurrentValues) > maxNumberOfValues:
            clusteredInstances = llmBasedCLuster(kbElement['canonical string'],relationName, knowledgeBaseID)
            for clusterFromClusteredInstances in clusteredInstances:
                clusterName = 'lmmBasedClusterNaming(clusterFromClusteredInstances)'
                generalizedConceptName = '' #TODO: add the generalization of the current entity
                addValueInverseAndTypes(clusterName, 'generalizations',generalizedConceptName,knowledgeBaseID,iteration, 'conceptSpliting',currentDate, '0.5','','', knowledgeBaseIDPath,fullyTrustedSources)


def checkPromotionToBelief_frequencyBased(kbElementproperties,kbElement,knowledgeBaseID,iteration,promotionCriteria):
    #This frequency-based promotion policy uses a simple (given) frequency threshold. If the "learned" piece of knowledge was "extracted" N times and N is larger than the frequencyThreshold, then that piece of knowledge is pro,oted to belief. 
    
    #go through all relations (keys) in the json file
    for relationName in kbElementproperties:
        MyLogger().log('checkPromotionToBelief_frequencyBased: Checking Relation:'+str(relationName))
        if relationName == 'generalizations':
            #go over all generalizations for the current entity/file
            #TODO: check whether we need a special case for "generalizations" or can we generalize with all the other relations and eliminate this "if clause"
            for generalization in kbElementproperties['generalizations']:
                #start with an empty list of seeds so that we can build the new list of instances that will be used in the prompts of the next iteration
                MyLogger().log('checkPromotionToBelief_frequencyBased: Checking the value: '+ str(generalization) +' Relation:'+str(relationName))
                listOfSeedEntities = []

                #test whether the instance should be used as seed for the next iteration or not
                beliefCandidate = kbElementproperties['generalizations'][generalization]
                if isBelief(beliefCandidate,promotionCriteria):
                    #set the seeds file name to be used to write the file
                    seedsFileName = 'data/kbs/' + str(knowledgeBaseID) + '/concepts/'+ str(iteration+1) + '/' + generalization+'.txt'                    
                    #accumulatedSeedsFileName = 'data/kbs/' + str(knowledgeBaseID) + '/concepts/' + str(iteration+1)+ '/' + generalization + '_accumulated.txt'
                    if os.path.exists(seedsFileName):
                        with open(seedsFileName, 'r') as seedsFile:
                            seedsFileContent = seedsFile.read()
                        listOfSeedEntities = ast.literal_eval(seedsFileContent)
                    else:
                        listOfSeedEntities = []
                    if kbElement.lower() not in listOfSeedEntities: #avoid duplicates. In a different strategy, it is possoble to allow multiple repetitions of the same seed and use that as a weight for the random sampling for the next iteration
                        listOfSeedEntities.append(kbElement.lower())
                    with open(seedsFileName, 'w') as seedsFile: #TODO: store all additions for a give iteration so that we can open and add them all together in a single acess to the file
                        seedsFile.write(str(listOfSeedEntities))
        else:
            #it is a relation different than 'generalizations' 
            if relationName != 'canonical string' and relationName != 'literal string' and '_inverse of ' not in relationName:
                for relationObject in kbElementproperties[relationName]:
                    #go over all other relation values
                    MyLogger().log('checkPromotionToBelief_frequencyBased: Checking the value: '+ str(relationObject) +' Relation:'+str(relationName))
                    #relationName = relationName
                    listOfSeedEntities = []
                    beliefCandidate = kbElementproperties[relationName][relationObject]
                    if isBelief(beliefCandidate,promotionCriteria):
                        relationSubject = kbElement.lower()
                        relationObject = relationObject.lower()
                        seedsFileName = 'data/kbs/' + str(knowledgeBaseID) + '/relations/'+ str(iteration+1) + '/' + relationName+'.txt'
                        if os.path.exists(seedsFileName):
                            with open(seedsFileName, 'r') as seedsFile:
                                seedsFileContent = seedsFile.read()
                            listOfSeedEntities = ast.literal_eval(seedsFileContent)
                        else:
                            listOfSeedEntities = []
                        pairToBeAdded = (relationSubject,relationObject)
                        if pairToBeAdded not in listOfSeedEntities: #avoid duplicates. In a different strategy, it is possoble to allow multiple repetitions of the same seed and use that as a weight for the random sampling for the next iteration
                            listOfSeedEntities.append((relationSubject,relationObject))
                        with open(seedsFileName, 'w') as seedsFile:
                            seedsFile.write(str(listOfSeedEntities))

def isBelief(beliefCandidate,promotionCriteria):
    """
    promotionCriteria is a list with 2 elements [promotionCriteria[0],promotionCriteria[1]]. The first element of the list
    defines the name of the criteria. The second element of the tuple is the parameter for the specific criteria, as follows:
        * promotionCriteria[0]=='simpleFrequency', then promotionCriteria[1]=='frequency threshold'
        * promotionCriteria[0]== ...
    """
    if promotionCriteria[0]=='simpleFrequency':
        if 'totalCount'in beliefCandidate:
            if beliefCandidate['totalCount'] > promotionCriteria[1]:
                return True
    return False


def probe_T5_ModelWithGenericPrompt(entityListPairs,promptTemplates,promptList,conceptName,numberOfExtractions,prober,db, iterationNumber,knowledgeBaseID):
    extractedNames = []
    totalNumberOfPairs = len(entityListPairs)
    currentPair = 0
    for pair in entityListPairs:
        lowercaseCannonicalExtractedName = 'none'
        currentPair += 1
        for promptTemplate in promptTemplates:
            MyLogger().log('Iteration: ' + str(iterationNumber)+ '_' + knowledgeBaseID + '. Processing pair ' + str(currentPair) + ' out of ' + str(totalNumberOfPairs) + ' for concept: ' + str(conceptName) + '. ')
            MyLogger().log(promptTemplate)
            promptList.append(promptTemplate)
            currentPrompt = promptTemplate
            #TODO:check if we are going over all templates for each pair
            prompt = eval(currentPrompt)
            MyLogger().log(prompt)
            try:
                extractedTokens = prober(prompt, topk=numberOfExtractions, max_new_tokens=50)
            except:
                MyLogger().log('Error in prompt: ' + prompt)
                continue
            extractedNames = []
            for extractedItem in extractedTokens['values']:
                #cannonicalExtractedName = filterPuctuationAtTheEndOfEntity(extractedItem['token'],spacyModel) #.strip().replace('.', '').replace('!', '').replace('?', '')
                #TODO: fix the problem when loading spacy and enable the line above and the  line below
                #extractedNames.append((cannonicalExtractedName,extractedItem['token']))
                cannonicalName = cannonicalEntityName(extractedItem['token'])
                if cannonicalName != '':
                    extractedNames.append((cannonicalName,extractedItem['token']))
                    
                    #add the extracted triple
                    addToJsonFile(cannonicalName,extractedItem['token'],promptTemplate,prompt,cannonicalName,'generalizations',conceptName,iterationNumber,knowledgeBaseID)
                    MyLogger().log(extractedItem['token'], end='; ')

            MyLogger().log("\n=====================================")
            for extractedName in set(extractedNames): #eliminate duplicates and add to the extractions
                lowercaseCannonicalExtractedName = str(extractedName[0]).lower()
    return extractedNames

def probe_vLLM_ModelWithGenericPrompt(entityListPairs,promptTemplates,promptList,conceptName,numberOfExtractions,prober,db, iterationNumber,knowledgeBaseID):
    extractedNames = []
    totalNumberOfPairs = len(entityListPairs)
    currentPair = 0
    for pair in entityListPairs:
        currentPair += 1
        for promptTemplate in promptTemplates:
            MyLogger().log('Iteration: ' + str(iterationNumber)+ '_' + knowledgeBaseID + '. Processing pair ' + str(currentPair) + ' out of ' + str(totalNumberOfPairs) + ' for concept: ' + str(conceptName) + '. ')
            MyLogger().log(promptTemplate)
            promptList.append(promptTemplate)
            currentPrompt = promptTemplate
            #TODO:check if we are going over all templates for each pair
            prompt = eval(currentPrompt)
            MyLogger().log(prompt)
            try:
                extractedTokens = prober(prompt, topk=numberOfExtractions, max_new_tokens=50)
            except:
                MyLogger().log('Error in prompt: ' + prompt)
                continue
            extractedNames = []
            for extractedItem in extractedTokens['values']:
                #cannonicalExtractedName = filterPuctuationAtTheEndOfEntity(extractedItem['token'],spacyModel) #.strip().replace('.', '').replace('!', '').replace('?', '')
                #TODO: fix the problem when loading spacy and enable the line above and the  line below
                #extractedNames.append((cannonicalExtractedName,extractedItem['token']))
                cannonicalName = cannonicalEntityName(extractedItem['token'])
                if cannonicalName != '':
                    extractedNames.append((cannonicalName,extractedItem['token']))
                    
                    #add the extracted triple
                    addToJsonFile(cannonicalName,extractedItem['token'],promptTemplate,prompt,cannonicalName,'generalizations',conceptName,iterationNumber,knowledgeBaseID)
                    MyLogger().log(extractedItem['token'], end='; ')

                    #add the inverse of the extracted triple
                    addToJsonFile(cannonicalName,extractedItem['token'],promptTemplate,prompt,cannonicalName,'generalizations',conceptName,iterationNumber,knowledgeBaseID)
                    MyLogger().log(extractedItem['token'], end='; ')

            MyLogger().log("\n=====================================")
            for extractedName in set(extractedNames): #eliminate duplicates and add to the extractions
                lowercaseCannonicalExtractedName = str(extractedName[0]).lower()
    return extractedNames

def probeModelWithRelationsGenericPrompt(relationInstancesListPairs,relationPromptTemplates,relationPromptList,relationName,numberOfExtractions,prober,db,iterationNumber,knowledgeBaseID):
    extractedNames = []
    totalNumberOfPairs = len(relationInstancesListPairs)
    currentPair = 0
    for pair in relationInstancesListPairs:
        currentPair += 1
        for promptTemplate in relationPromptTemplates:
            MyLogger().log('Iteration: ' + str(iterationNumber)+ '_' + knowledgeBaseID + '. Processing pair ' + str(currentPair) + ' out of ' + str(totalNumberOfPairs) + ' for relation: ' + str(relationName) + '. ')
            MyLogger().log(promptTemplate)
            relationPromptList.append(promptTemplate)
            currentPrompt = promptTemplate
            #TODO:check if we are going over all templates for each pair
            prompt = eval(currentPrompt[0])
            MyLogger().log(prompt)
            try:
                extractedTokens = prober(prompt, topk=numberOfExtractions, max_new_tokens=50)
            except:
                MyLogger().log('Error in prompt: ' + prompt)
                continue
            extractedNames = []
            for extractedItem in extractedTokens['values']:
                #cannonicalExtractedName = filterPuctuationAtTheEndOfEntity(extractedItem['token'],spacyModel) #.strip().replace('.', '').replace('!', '').replace('?', '')
                #TODO: fix the problem when loading spacy and enable the line above and the  line below
                #extractedNames.append((cannonicalExtractedName,extractedItem['token']))
                extractedLiteralString = extractedItem['token']
                cannonicalName = cannonicalEntityName(extractedLiteralString)
                if cannonicalName != '':
                    if currentPrompt[1][0] == '[MASK]':
                        extractedPair = ((cannonicalName,extractedItem['token']),eval(currentPrompt[1][1]))
                        tripletSubject = cannonicalName
                        tripletObject = eval(currentPrompt[1][1])
                        #add the subject as an intance for the object type (concept). If "Apple" is extracted in the triple "Apple is located in Cupertino", then, "Apple" must be added as "Apple generalization companies"
                        tripletSubjectType = relationName.split('_')[0]
                        addToJsonFile(str(cannonicalName),str(extractedLiteralString),str(promptTemplate),prompt,tripletSubject,'generalizations',tripletSubjectType,iterationNumber,knowledgeBaseID)
                    else:
                        extractedPair = (eval(currentPrompt[1][0]),(cannonicalName,extractedItem['token']))
                        tripletSubject = eval(currentPrompt[1][0])
                        tripletObject = cannonicalName
                        #add the object as an intance for the object type (concept). If "Cupertino" is extracted in the triple "Apple is located in Cupertino", then, "Cupertino" must be added as "Cupertno generalization cities"
                        tripletObjectType = relationName.split('_')[2]
                        addToJsonFile(str(cannonicalName),str(extractedLiteralString),str(promptTemplate),prompt,tripletObject,'generalizations',tripletObjectType,iterationNumber,knowledgeBaseID)
                    extractedNames.append(extractedPair)
                    #add the subject-predicate-object triplet itself. If the triplet "Apple is located in Cupertino" is extracted, then, the triplet will be added to "apple.json" as is_located_in:cupertino. And will be added to "cupertino.json" as inverseOfIs_located_in:apple
                    addToJsonFile(str(cannonicalName),str(extractedLiteralString),str(promptTemplate),prompt,tripletSubject,relationName,tripletObject,iterationNumber,knowledgeBaseID)
                    MyLogger().log(extractedItem['token'], end='; \n')
            MyLogger().log("\n=====================================")
            for extractedName in set(extractedNames): #eliminate duplicates and add to the extractions
                lowercaseCannonicalExtractedName = str(extractedName[0]).lower()
    return extractedNames

def updateJson(dictFromJson,promptTemplate,prompt,literalString,tripletSubject,tripletPredicate,tripletObject,globalIteration,source):
    if tripletPredicate not in dictFromJson:
        dictFromJson[tripletPredicate] = {}
    if tripletObject not in dictFromJson[tripletPredicate]:
        dictFromJson[tripletPredicate][tripletObject] = {}
        dictFromJson[tripletPredicate][tripletObject]['totalCount'] = 0
    dictFromJson[tripletPredicate][tripletObject]['totalCount'] += 1
    if 'promptTemplates' not in dictFromJson[tripletPredicate][tripletObject]:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'] = {}
    if promptTemplate not in dictFromJson[tripletPredicate][tripletObject]['promptTemplates']:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate] = {}
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['totalCountPerPrompTemplate'] = 0
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['totalCountPerPrompTemplate'] += 1
    if 'prompts' not in dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'] = {}
    if prompt not in dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts']:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt] = {}
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['totalCountPerPromp'] = 0
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['totalCountPerPromp'] += 1
    if 'literalStrings' not in dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'] = {}
    if literalString not in dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings']:
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString] = {}
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['count'] = 0
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['date'] = ''
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['source'] = ''
        dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['iteration'] = ''
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['count'] += 1
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['date'] = str(dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['date']) + ' | ' + str(datetime.now())
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['source'] = str(dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['source']) + ' | ' + source
    dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['iteration'] = str(dictFromJson[tripletPredicate][tripletObject]['promptTemplates'][promptTemplate]['prompts'][prompt]['literalStrings'][literalString]['iteration']) + ' | ' + str(globalIteration)
   
    return dictFromJson

def createCopyOfSeedsForNextIteration(basePath,iteration):
    splittedBasePath = basePath.split('/')
    pathToBeCreated = ''
    for dirItem in splittedBasePath:
        if dirItem != '':
            pathToBeCreated = pathToBeCreated + dirItem + '/'
            if not os.path.isdir(pathToBeCreated):
                os.makedirs(pathToBeCreated)
    return pathToBeCreated
def createPath(basePath):
    splittedBasePath = basePath.split('/')
    pathToBeCreated = ''
    for dirItem in splittedBasePath:
        if dirItem != '':
            pathToBeCreated = pathToBeCreated + dirItem + '/'
            if not os.path.isdir(pathToBeCreated):
                os.makedirs(pathToBeCreated)
    return pathToBeCreated

def buildPath(basePath):
    if not os.path.isdir(basePath):
        splittedBasePath = basePath.split('/')
        pathToBeCreated = ''
        for dirItem in splittedBasePath:
            if dirItem != '':
                pathToBeCreated = pathToBeCreated + dirItem + '/'
                if not os.path.isdir(pathToBeCreated):
                    os.makedirs(pathToBeCreated)


def createPhysicalFile(cannonicalName, basePath):
    if not os.path.isdir(basePath):
        os.makedirs(basePath)
    if len(cannonicalName) == 1:
        tempPath = basePath + cannonicalName[0].lower()
        if not os.path.isdir(tempPath):
            os.makedirs(tempPath)
        pathName = basePath + cannonicalName[0].lower() + '/' + cannonicalName.lower() + '.json'
    else:
        if len(cannonicalName) == 2:
            tempPath = basePath + cannonicalName[0]
            if not os.path.isdir(tempPath.lower()):
                os.makedirs(tempPath.lower())
            tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1]
            if not os.path.isdir(tempPath.lower()):
                os.makedirs(tempPath.lower())
            pathName = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName + '.json'
        else:
            if len(cannonicalName) == 3:
                tempPath = basePath + cannonicalName[0]
                if not os.path.isdir(tempPath.lower()):
                    os.makedirs(tempPath.lower())
                tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1]
                if not os.path.isdir(tempPath.lower()):
                    os.makedirs(tempPath.lower())
                tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2]
                if not os.path.isdir(tempPath.lower()):
                    os.makedirs(tempPath.lower())
                pathName = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName + '.json'
            else:
                if len(cannonicalName) == 4:
                    tempPath = basePath + cannonicalName[0]
                    if not os.path.isdir(tempPath.lower()):
                        os.makedirs(tempPath.lower())
                    tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1]
                    if not os.path.isdir(tempPath.lower()):
                        os.makedirs(tempPath.lower())
                    tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2]
                    if not os.path.isdir(tempPath.lower()):
                        os.makedirs(tempPath.lower())
                    tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName[3]
                    if not os.path.isdir(tempPath.lower()):
                        os.makedirs(tempPath.lower())
                    pathName = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName[3]+ '/' + cannonicalName + '.json'
                else:
                    if len(cannonicalName) > 1: # empty strings should be filtered out
                        tempPath = basePath + cannonicalName[0]
                        if not os.path.isdir(tempPath.lower()):
                            os.makedirs(tempPath.lower())
                        tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1]
                        if not os.path.isdir(tempPath.lower()):
                            os.makedirs(tempPath.lower())
                        tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2]
                        if not os.path.isdir(tempPath.lower()):
                            os.makedirs(tempPath.lower())
                        tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName[3]
                        if not os.path.isdir(tempPath.lower()):
                            os.makedirs(tempPath.lower())
                        tempPath = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName[3]+ '/' + cannonicalName[4]
                        if not os.path.isdir(tempPath.lower()):
                            os.makedirs(tempPath.lower())
                        pathName = basePath + cannonicalName[0] + '/' + cannonicalName[1] + '/' + cannonicalName[2] + '/' + cannonicalName[3]+ '/' + cannonicalName[4] + '/' + cannonicalName + '.json'
                    else:
                        MyLogger().log('Skipping because the entity name is empty!',level="ERROR")
                        pathName = 'ERROR'
    return pathName

def addToTheFile(jsonFileName, promptTemplate, prompt,literalString, tripletSubject, tripletPredicate, tripletObject,iterationNumber, source):
    dictFromJson = {}
    if os.path.exists(jsonFileName):
        with open(jsonFileName, 'r') as jsonFile:
            dictFromJson = json.load(jsonFile)
        dictFromJson = updateJson(dictFromJson,promptTemplate,prompt,literalString,tripletSubject,tripletPredicate,tripletObject,iterationNumber,source)           
    else:
        dictFromJson = updateJson({},promptTemplate,prompt,literalString,tripletSubject, tripletPredicate, tripletObject, iterationNumber, source)
        if tripletPredicate == 'pairsOfInstances': #check if this is the creation of a relation entity, and if it is, then add the relation properties
            dictFromJson = getPropertiesOfRelation(jsonFileName, tripletPredicate, dictFromJson)
            dictFromJson = updateJson(dictFromJson,promptTemplate,prompt,literalString,tripletSubject, tripletPredicate, tripletObject, iterationNumber, source)
        else:
            dictFromJson = updateJson({},promptTemplate,prompt,literalString,tripletSubject, tripletPredicate, tripletObject, iterationNumber, source)

    with open(jsonFileName, 'w') as jsonFile:
        json.dump(dictFromJson,jsonFile, indent=4)


def  getPropertiesOfRelation(jsonFileName,tripletPredicate, dictFromJson):
    #inverseName = getTheInverse(tripletPredicate)
    inverseName = tripletPredicate + '_inverse'
    seedInstances = getInitialPairsOfConceptSeeds(tripletPredicate, jsonFileName.split('/kb/')[0]+'/kb/')
    dictFromJson['inverse'] = inverseName
    dictFromJson['description'] = 'relation'
    dictFromJson['seedInstances'] = seedInstances
    setOfProperties = llmRelationPropertiesExtraction(tripletPredicate)
    for newProperty in setOfProperties:
        dictFromJson[newProperty] = setOfProperties[newProperty]
    return dictFromJson 
def llmRelationPropertiesExtraction(tripletPredicate):
    #TODO: use LLM to add properties
    setOfProperties = {}
    pass
    return setOfProperties

def addToJsonFile(literalString,promptTemplate,prompt, tripletSubject,tripletPredicate, tripletObject, iterationNumber,knowledgeBaseID, modelName):
    jsonFileName_subject = createPhysicalFile(tripletSubject.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    jsonFileName_predicate = createPhysicalFile(tripletPredicate.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    contextualTripletPredicate = tripletPredicate
    if len(tripletPredicate.split('_'))==1:
        contextualTripletPredicate = str(tripletSubject) + '_' + str(contextualTripletPredicate) + '_' + str(tripletObject)
    #inversePredicateName = getTheInverse(contextualTripletPredicate,modelName)
    inversePredicateName = str(tripletPredicate) + '_inverse' 
    jsonFileName_inversePredicate = createPhysicalFile(inversePredicateName.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    subjectType = tripletPredicate.split('_')[0]
    jsonFileName_subjectType = createPhysicalFile(subjectType.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    objectType = tripletPredicate.split('_')[2]
    jsonFileName_objectType = createPhysicalFile(objectType.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    jsonFileName_object = createPhysicalFile(tripletObject.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    #Add all elements of the triple to the physical json file
    
    #Adding the subject type generalization-specialization, if not yet created
    #addToTheFile(jsonFileName_subjectType, promptTemplate, prompt,subjectType, tripletObject, 'specializations', tripletSubject,iterationNumber, modelName)
    #Adding the subject and object generalizations and specializations
    if tripletPredicate != 'generalizations' and tripletPredicate != 'specializations':
        addToTheFile(jsonFileName_subject, promptTemplate, prompt,subjectType, tripletSubject, 'generalizations', subjectType,iterationNumber, modelName)
        addToTheFile(jsonFileName_object, promptTemplate, prompt,objectType, tripletObject, 'generalizations', objectType,iterationNumber, modelName)
        addToTheFile(jsonFileName_subjectType, promptTemplate, prompt,subjectType, subjectType, 'specializations', tripletSubject,iterationNumber, modelName)
        addToTheFile(jsonFileName_objectType, promptTemplate, prompt,objectType, objectType, 'specializations', tripletObject,iterationNumber, modelName)

    #Adding the subject
    addToTheFile(jsonFileName_subject, promptTemplate, prompt,tripletObject, tripletSubject, tripletPredicate, tripletObject,iterationNumber, modelName)

    #Adding the predicate
    addToTheFile(jsonFileName_predicate, promptTemplate, prompt,str((tripletSubject, tripletObject)), tripletPredicate, 'pairsOfInstances' ,str((tripletSubject, tripletObject)),iterationNumber, modelName)
    addToTheFile(jsonFileName_inversePredicate, promptTemplate, prompt,str((tripletObject, tripletSubject)), inversePredicateName,'pairsOfInstances', str((tripletObject, tripletSubject)),iterationNumber, modelName)

    #Adding the oject type
    #addToTheFile(jsonFileName_objectType, promptTemplate, prompt,objectType, inversePredicateName, tripletObject, tripletSubject,iterationNumber, modelName)
    #Adding the oject
    addToTheFile(jsonFileName_object, promptTemplate, prompt,tripletSubject, tripletObject, inversePredicateName, tripletSubject,iterationNumber, modelName)


def cannonicalEntityName(originalEntityName):
    cannonicalName = originalEntityName.split(',')[0].split('.')[0].split('/')[0]
    return cannonicalName

def addItemsFromLMtoList(lmOutput,initialList,minimumSize,clean):
    if clean:
        for lmOutputItem in lmOutput['values']:
            lmOutputItemName = lmOutputItem['token'].strip().replace('.', ' ').replace('.', ' ').replace(',', ' ')
            if not filterNoiseAndSingleWord(lmOutputItemName,minimumSize):
                initialList.append(lmOutputItemName.strip())
    else:
        for lmOutputItem in lmOutput['values']:
            lmOutputItemName = lmOutputItem['token']
            initialList.append(lmOutputItemName.strip())
    return initialList


def filterNoiseAndSingleWord(lmOutputItemName,minimumSize):
    """
    TODO
    """    
    pass

"""
def addExtractionsToTheDB(tokens, prompt_template,prompt,pair,db):
    for token in tokens:
        data = {
            "cannonical_value": token[0],
            "literal_string": token[1],
            "updated_at": datetime.now(),
        }
        set_on_insert = {
                "created_at": datetime.now(),
                "status": "new",
        }
        # collection.update_one({"value": token['token']}, {"$set": data, "$setOnInsert": set_on_insert, "$addToSet":{"source.prompt_template."+prompt_template['prompt']:query_prompt,"disposition": "new"} ,"$inc":{"count":1}}, upsert=True)
        db.upsert_entity(prompt_template, token[0], prompt, pair)
        print(token, end='; ')
    print("\n=====================================")     
"""

def fetchPromptsFromFile(prompts):
    promptList = []
    #with open(promptsFileName, 'r' ) as promptsFile: #reads a json file containing a list of prompt dictionaries
                                    #with keys: "prompt", "promptType" (can be "seedTemplate" or "learnedTemplate")
                                    #and "confidenceScore" (the confidence the model/system has that the prompt is
                                    # precise in extracting good candidates)
    #    promptDictList = json.load(promptsFile)
    for prompt in prompts:
        promptList.append(prompt['prompt'])
    return promptList

def fetchRelationPromptsFromFile(prompts):
    promptList = []
    #with open(promptsFileName, 'r' ) as promptsFile: #reads a json file containing a list of prompt dictionaries
                                    #with keys: "prompt", "promptType" (can be "seedTemplate" or "learnedTemplate")
                                    #and "confidenceScore" (the confidence the model/system has that the prompt is
                                    # precise in extracting good candidates)
    #    promptDictList = json.load(promptsFile)
    for prompt in prompts:
        assembledPrompt = prompt['prompt'].replace('relationName',prompt['relationName'])
        assembledPrompt = assembledPrompt.replace('subjectType',prompt['subjectType'])
        assembledPrompt = assembledPrompt.replace('objectType',prompt['objectType'])
        if prompt['MASK_TYPE'] == 'objectType':
            promptList.append(((assembledPrompt),(prompt['MASK_PAIR'],'[MASK]')))
        else:
            if prompt['MASK_TYPE'] == 'subjectType':
                promptList.append(((assembledPrompt),('[MASK]',prompt['MASK_PAIR'])))
    return promptList

def get_entity_from_prompt(prompt):
    return re.findall('\[([A-z]*)\]', prompt)

def get_limited_entities(db, entity_type, query={}, limit=0):
    collection = db[entity_type]
    entity = list(collection.aggregate([{"$match":query},{"$sample": {"size": limit}}]))
    return [e['value'] for e in entity]

def readListOfEntitiesFromTxtFile(txtInputFile):
    try:
        with open(txtInputFile, 'r') as inputFile:
            fileContent = inputFile.read()
            listOfEntities = ast.literal_eval(fileContent)
    except:
        return [] # if the file can't be read, just return the empty list       
    return listOfEntities

def getConcepts(conceptsPath,iteration):
    listOfConcepts = []
    for file in os.listdir(conceptsPath):
        if not file.endswith('_accumulated.txt'):
            if len(file.split('_')) == 1:
                listOfConcepts.append(file.split('.txt')[0])
    return listOfConcepts

def seedsFromOnet(knowledgeBaseID, ingestionIteration, startOfCurrentIterationDate, confidence,knowledgeBaseIDPath,oNetPath,fullyTrustedSources):
    #ingest skills from O*Net
    ingestionFileFileName = str(oNetPath)+'Skills-NoDup.txt'
    ingestionConcept = 'skills'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest abilities from O*Net
    ingestionFileFileName = str(oNetPath)+'Abilities-NoDup.txt'
    ingestionConcept = 'abilities'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest knowledge from O*Net
    ingestionFileFileName = str(oNetPath)+'Knowledge-NoDup.txt'
    ingestionConcept = 'knowledge'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest Work Activities from O*Net
    ingestionFileFileName = str(oNetPath)+'Work Activities-NoDup.txt'
    ingestionConcept = 'work activities'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)
    
    #ingest Technology Skills from O*Net
    ingestionFileFileName = str(oNetPath)+'Technology Skills.txt'
    ingestionConcept = 'technology skills'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestExamplesFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest Technology Skills from O*Net
    ingestionFileFileName = str(oNetPath)+'Tools Used.txt'
    ingestionConcept = 'tools'
    provenanceInfo = ingestionFileFileName#+'||concept||'+str(ingestionConcept)
    ingestExamplesFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest Occupation Descriptions from O*Net
    ingestionFileFileName = str(oNetPath)+'Occupation Data.txt'
    ingestionConcept = 'descriptions'
    provenanceInfo = ingestionFileFileName+'||concept||'+str(ingestionConcept)
    ingestOccupationDescriptionsFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

    #ingest relational pairs from O*net
    ingestionFileFileName = str(oNetPath)+'Abilities to Work Activities.txt'
    relationPredicate = 'abilities_required by_work activities'
    ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID, knowledgeBaseIDPath,fullyTrustedSources)

    ingestionFileFileName = str(oNetPath)+'Abilities to Work Context.txt'
    relationPredicate = 'abilities_required by_work context'
    ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID, knowledgeBaseIDPath,fullyTrustedSources)

    ingestionFileFileName = str(oNetPath)+'Skills to Work Activities.txt'
    relationPredicate = 'skills_required by_work activities'
    ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID, knowledgeBaseIDPath,fullyTrustedSources)
 
    ingestionFileFileName = str(oNetPath)+'Skills to Work Context.txt'
    relationPredicate = 'skills_required by_work context'
    ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID, knowledgeBaseIDPath,fullyTrustedSources)

    ingestionFileFileName = str(oNetPath)+'Technology Skills.txt'
    relationPredicate = 'skills_required by_work context'
    ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID, knowledgeBaseIDPath,fullyTrustedSources)


def ingestFileFromOnet(concept,ingestionFileFileName, iteration,ingestionDate, knowledgeBaseID, confidence,provenanceInfo, knowledgeBaseIDPath,fullyTrustedSources):
    """
    This is a ingestion function hard-coded to the format present in O*Net files: "Skills.txt", which are tsv files with the header row as:
    O*NET-SOC Code	Element ID	Element Name	Scale ID ...
    relative paths: data/onet/db_29_1_text/Skills.txt
                    data/onet/db_29_1_text/Knowledge.txt
    """
    #ingestionFileFileName = 'data/onet/db_29_1_text/Skills.txt'
    ingestionFileDataframe = pd.read_csv(ingestionFileFileName, sep='\t')
    for index, row in ingestionFileDataframe.iterrows():
        #add the generalization
        relationSubject = str(row[ingestionFileDataframe.columns[2]])
        relationPredicate = 'generalizations'
        relationObject = concept #'skills'
        sourceOfInformation = 'O*Net'
        fullProvenance = provenanceInfo+'||'+str(relationSubject)+ '||'+str(relationPredicate+'||'+str(relationObject))
        confidence = '1'
        addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID,iteration,sourceOfInformation,ingestionDate,confidence,fullProvenance, knowledgeBaseIDPath,fullyTrustedSources)
        """
        addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID)
        print('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
        #add the source, iteration and date of the generalization
        relationSubject = str(row[ingestionFileDataframe.columns[2]]+'||generalizations||'+str(concept)+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate))
        relationPredicate = 'confidence'
        relationObject = confidence
        addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID)

        #add the specialistion
        relationSubject = concept #'skills'
        relationPredicate = 'specializations'
        relationObject = str(row[ingestionFileDataframe.columns[2]])
        addValue(relationSubject,relationPredicate,relationObject, knowledgeBaseID)
        print('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
        #add the source, iteration and date of the specialization
        relationSubject = str(concept)+'||specializations||'+str(row[ingestionFileDataframe.columns[2]])+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate)
        relationPredicate = 'confidence'
        relationObject = confidence
        addValue(relationSubject,relationPredicate,relationObject, knowledgeBaseID)

        #add the generalizations for the 
        """

def ingestExamplesFileFromOnet(concept,ingestionFileFileName, iteration,ingestionDate, knowledgeBaseID, confidence,provenanceInfo, knowledgeBaseIDPath,fullyTrustedSources):
    """
    This is a ingestion function hard-coded to the format present in O*Net files: "Tecnology Skills.txt", which are tsv files with the header row as:
    O*NET-SOC Code	Example	Commodity Code	Commodity Title	Hot Technology	In Demand
    11-1011.00	Adobe Acrobat	43232202	Document management software	Y	N
    11-1011.00	AdSense Tracker	43232306	Data base user interface and query software	N	N
    11-1011.00	Atlassian JIRA	43232201	Content workflow software	Y	N
    
    relative paths: data/onet/db_29_1_text/Technology Skills.txt
                    data/onet/db_29_1_text/Tools Used.txt
    """
    ingestionFileDataframe = pd.read_csv(ingestionFileFileName, sep='\t')
    for index, row in ingestionFileDataframe.iterrows():
        #add the generalization
        relationSubject = str(row[ingestionFileDataframe.columns[3]])
        relationPredicate = 'generalizations'
        relationObject = concept
        sourceOfInformation = 'O*Net'
        fullProvenance = provenanceInfo+'||'+str(relationSubject)+ '||'+str(relationPredicate+'||'+str(relationObject))
        confidence = '1'
        addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID,iteration,sourceOfInformation,ingestionDate,confidence,fullProvenance, knowledgeBaseIDPath,fullyTrustedSources)
        relationObject = str(row[ingestionFileDataframe.columns[3]])
        relationSubject = str(row[ingestionFileDataframe.columns[1]])
        addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID,iteration,sourceOfInformation,ingestionDate,confidence,fullProvenance, knowledgeBaseIDPath,fullyTrustedSources)


def ingestFromTriples(triplesFileName, knowledgeBaseID, currentIteration, sourceOfInformation, currentDate, confidence, provenanceInfo, knowledgeBaseIDPath,fullyTrustedSources):
    numberOfIngestedTriples = 0
    try: 
        with open(triplesFileName, 'r') as triplesFile:
            MyLogger().log('ingesting the triples from File'+str(triplesFileName), level='INFO')
            for line in triplesFile:
                fullProvenanceInfo = provenanceInfo
                splittedLine = line.replace('\n','').split('\t')
                if len(splittedLine)==3:
                    provenance = splittedLine[0]+'||'+splittedLine[1]+'||'+splittedLine[2]
                    fullProvenanceInfo = provenanceInfo +'||' + provenance
                    addValueInverseAndTypes(splittedLine[0],splittedLine[1], splittedLine[2], knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,fullProvenanceInfo, knowledgeBaseIDPath,fullyTrustedSources)
                    numberOfIngestedTriples += 1
                    MyLogger().log('\n Triples ingested so far: ' + str(numberOfIngestedTriples))
    except FileNotFoundError:
        MyLogger().log(str(triplesFileName)+ ' not found.', level="ERROR")
    MyLogger().log('\n\n'+ str(numberOfIngestedTriples)+ ' triples ingested.\n')


def ingestRelationalPairFromOnet(ingestionFileFileName, relationPredicate, iteration,ingestionDate, knowledgeBaseID, knowledgeBasePath,fullyTrustedSources):
    """
    This is an ingestion template for input files coming from Onet conceptToConcept files. Example: 'Skills to Work Activities.txt'.
    The format of the input data is always a tsv file with four columns, where the 1st column is the 1st argument ID, the second column is the 1st argument label, the 3rd column is the second argument ID and the 4th and last column is the 2nd argument label. Based on the example we would have:
    
    Skills Element ID	Skills Element Name	Work Activities Element ID	Work Activities Element Name
    2.A.1.a	Reading Comprehension	4.A.1.a.1	Getting Information
    2.A.1.a	Reading Comprehension	4.A.1.a.2	Monitoring Processes, Materials, or Surroundings
    2.A.1.a	Reading Comprehension	4.A.1.b.1	Identifying Objects, Actions, and Events
    ...
    """
    sourceOfInformation = 'O*Net'
    provenanceInfo = 'instances file||'+ingestionFileFileName
    ingestionFileDataframe = pd.read_csv(ingestionFileFileName, sep='\t')
    confidence = '1'
    for index, row in ingestionFileDataframe.iterrows():

        #add the relation "descriptions"
        relationSubject = str(row[ingestionFileDataframe.columns[1]])
        relationObject = str(row[ingestionFileDataframe.columns[3]])
        addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID,iteration,sourceOfInformation,ingestionDate,confidence,provenanceInfo,knowledgeBasePath,fullyTrustedSources)


def ingestOccupationDescriptionsFromOnet(concept,ingestionFileFileName, iteration,ingestionDate, knowledgeBaseID, confidence, knowledgeBasePath,fullyTrustedSources):
    """
    This is a ingestion function hard-coded to the format present in O*Net file Occupation Data.txt, 
    which are tsv files with the header row as:
    O*NET-SOC Code	Title	Description
    relative paths: data/onet/db_29_1_text/Occupation Data.txt
    """
    sourceOfInformation = 'O*Net'
    provenanceInfo = 'instances file||'+ingestionFileFileName
    ingestionFileFileName = 'data/onet/db_29_1_text/Occupation Data.txt'
    ingestionFileDataframe = pd.read_csv(ingestionFileFileName, sep='\t')
    
    for index, row in ingestionFileDataframe.iterrows():

        #add the relation "descriptions"
        relationSubject = str(row[ingestionFileDataframe.columns[1]])
        relationPredicate = 'occupations_has description_descriptions'
        relationObject = str(row[ingestionFileDataframe.columns[2]])
        valueAdded = addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID, iteration,sourceOfInformation,ingestionDate,confidence,provenanceInfo,knowledgeBasePath,fullyTrustedSources)
        if valueAdded == None:
            MyLogger().log('Triple is malformed and was not added: '+str(relationSubject) + '\t' + str(relationPredicate) + '\t'+ str(relationObject))
"""
        else:
            
            MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
            #add the source, iteration and date of the generalization
            relationSubject = str(row[ingestionFileDataframe.columns[1]]+'||occupations_has description_descriptions||'+str(concept)+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate))
            relationPredicate = 'confidence'
            relationObject = confidence
            addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBasePath)

            #add the inverse
            relationSubject = str(row[ingestionFileDataframe.columns[2]])
            relationPredicate = 'descriptions_inverse of has descriptions_occupations'
            relationObject = str(row[ingestionFileDataframe.columns[1]])
            addValue(relationSubject,relationPredicate,relationObject, knowledgeBaseID)
            MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
            #add the source, iteration and date of the specialization
            relationSubject = str(concept)+'||descriptions_inverse of has descriptions_occupations||'+str(row[ingestionFileDataframe.columns[2]])+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate)
            relationPredicate = 'confidence'
            relationObject = confidence
            addValue(relationSubject,relationPredicate,relationObject, knowledgeBaseID, knowledgeBasePath)

            #add the generalizations for the occupation
            relationSubject = str(row[ingestionFileDataframe.columns[1]])
            relationPredicate = 'generalizations'
            relationObject = 'occupations'
            addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBasePath)
            MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
            #add the source, iteration and date of the generalization
            relationSubject = str(row[ingestionFileDataframe.columns[2]]+'||generalizations||'+str(concept)+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate))
            relationPredicate = 'confidence'
            relationObject = confidence
            addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBasePath)

            #add the occupations specialization
            relationSubject = str(row[ingestionFileDataframe.columns[1]])
            relationPredicate = 'generalizations'
            relationObject = 'occupations'
            addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBasePath)
            MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + str(relationPredicate) + ', ' + str(relationObject))
            #add the source, iteration and date of the generalization
            relationSubject = str(row[ingestionFileDataframe.columns[2]]+'||generalizations||'+str(concept)+'||iteration||'+str(iteration)+'||source||O*Net||date||'+str(ingestionDate))
            relationPredicate = 'confidence'
            relationObject = confidence
            addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBasePath)
"""

def ingestRelationsFromSchemaDotOrg(schemaDotOrgFileName):
    pass


def ingestTriplesBatchFromFile(triplesFileName, iterationNumber, source, knowledgeBaseID):
    """ DEPRECATED"""
    promptTemplate = 'no prompt'
    prompt = 'no prompt: extracted from external source'
    triplesDataframe = pd.read_csv(triplesFileName, sep='\t')
    relationName = triplesFileName.split('/')[-1].replace('.tsv', '')
    relationSubjectType = triplesFileName.split('_')[0].split('/')
    relationObjectType = triplesFileName.split('_')[2].replace('.tsv', '')
    triplesDataframe = triplesDataframe.reset_index()  # make sure indexes pair with number of rows
    for index, row in triplesDataframe.iterrows():
        relationSubject = row[triplesDataframe.columns[1]]
        relationObject = row[triplesDataframe.columns[2]]
        #add the extracted triple
        addToJsonFile(relationSubject,promptTemplate,prompt,relationSubject,relationName,relationObject,iterationNumber,knowledgeBaseID, source)
        MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + str(relationName) + ', ' + str(relationObject))
        #the inverse of the extracted triple was already added in the 'addToJsonFile' function
        #relationNameInverse = relationName + '_inverse'
        #addToJsonFile(relationObject,promptTemplate,prompt,relationObject,relationNameInverse,relationSubject,iterationNumber,knowledgeBaseID, source)
        #print('Triple Added: ' + str(relationObject) + ', ' + str(relationNameInverse) + ', ' + str(relationSubject))


def getRelationsFromInitialDirectory(relationsPath):
    listOfRelations = []
    for fileName in os.listdir(relationsPath):
        if fileName.endswith('.txt'):
            if len(fileName.split('_'))==3:
                listOfRelations.append(fileName.replace('.txt',''))
    return listOfRelations

def getRelationsForConcept(concept, relationsPath):
    listOfRelations = []
    for file in os.listdir(relationsPath):
        if not file.endswith('_accumulated.txt'):
            if file.startswith(concept) and file !=  str(concept+".txt"):
                listOfRelations.append(file)
    return listOfRelations

def getInitialConceptSeeds(concept, iteration, conceptsPath):
    initialList = []
    seedsFileName = conceptsPath + str(concept) + '.txt'
    MyLogger().log(seedsFileName)
    initialList = readListOfEntitiesFromTxtFile(seedsFileName)
    return initialList

def getInitialPairsOfConceptSeeds(relationToBePopulated, relationsPath):
    initialList = []
    seedPairsFileName = relationsPath + str(relationToBePopulated)
    MyLogger().log(seedPairsFileName)
    initialList = readListOfEntitiesFromTxtFile(seedPairsFileName)
    return initialList


def getTheInverse(originalRelation, modelName): #TODO: add the inverse name based on LLM
    if originalRelation == 'generalizations':
        return 'specializations'
    if os.path.exists('inverseRelations.json'):
        with open('inverseRelations.json', 'r') as jsonFile:
            dictFromJson = json.load(jsonFile)
        return dictFromJson.setdefault(originalRelation,'inverseOf'+originalRelation)
    else:
        llmInverse = llmBasedInverseGeneration(originalRelation, modelName)
    return 'inverseOf'+originalRelation

def llmBasedInverseGeneration(relationName,modelName):
    #TODO: some prompt engineering to guarantee that the LLM generates correct inverse name
    with open('prompts/prompt_inverse_vllm.json') as prompt_inverseFile:
        promptFullTemplate = json.load(prompt_inverseFile)
        subjectType = eval(promptFullTemplate[0]['subjectType'])
        objectType = eval(promptFullTemplate[0]['objectType'])
        relationName = eval(promptFullTemplate[0]['relationName'])
        prompt = eval(promptFullTemplate[0]['prompt'])
    MyLogger().log(prompt)
    try:
        extractedTokens, modelName = probevLLM(prompt)
    except:
        MyLogger().log('Error in prompt: ' + prompt)
    extractedRelationName = extractedTokens[0].split('be: "')[1].split('"')[0]
    inverseRelation = objectType+'_'+extractedRelationName+'_'+subjectType
    return inverseRelation

def probevLLM(promptForVllm):
    prompts = '[vLLMlibTemp.tokenizer.apply_chat_template([{"role": "user", "content": promptForVllm}], tokenize=False),vLLMlibTemp.tokenizer.apply_chat_template([{"role": "user", "content": promptForVllm}], tokenize=False)]' #TODO: replace with an api call
    #Just test the vLLM
    #vLLM.run('test_inputs_v0.jsonl', 'test_outputs_v1.jsonl','meta-llama/Llama-2-7b-chat-hf',0.,1.0,128,None,'model_input')
    responses = 'vLLMlibTemp.llm.generate(prompts, sampling_params=vLLMlibTemp.sampling_params)'#TODO:replace with an api call
    listOfGeneratedOutputs = []
    for llmresponse in responses:
        for llmoutput in llmresponse.outputs:
            MyLogger().log(llmoutput.text.strip())
            listOfGeneratedOutputs.append(llmoutput.text.strip())
    return listOfGeneratedOutputs, 'vLLMlibTemp.llm.llm_engine.model_config.model' #TODO: replace with an api call

def updateJsonWithAutomaticInverse(dictFromJson,promptTemplate,prompt,literalString,tripletSubject,tripletPredicate,tripletObject,globalIteration,source):#dictFromJson,tripletSubject,relation,tripletObject,globalIteration,source):
    #calls the regular "updateJSON" and then call it again switching tripletSubject and tripletObject and use an inverse name for the relation
    updateJson(dictFromJson,promptTemplate,prompt,literalString,tripletSubject,tripletPredicate,tripletObject,globalIteration,source)
    tripletPredicateInverse = getTheInverse(tripletPredicate)
    if tripletPredicate == 'generalizations':
        updateJson(dictFromJson,promptTemplate,prompt,literalString,tripletSubject,'specializations',tripletObject,globalIteration,source)#dictFromJson,tripletObject,'specializations',tripletSubject,globalIteration,source)
    else:
        updateJson(dictFromJson,promptTemplate,prompt,tripletObject,tripletObject,tripletPredicateInverse,tripletSubject,globalIteration,source)#dictFromJson,tripletObject,inverseRelation,tripletSubject,globalIteration,source)
    return dictFromJson


def jsonFileNameCanonicalization(entityName, knowledgeBaseID):
    fullPathFileName = './data/kbs/' + knowledgeBaseID.lower() + '/kb/' + entityName.lower().replace(' ','_').replace('/','_')+'.json'
    return fullPathFileName


def getFileNameFromEntity(entity):
    entityInListFormat = entity.split('||') #entity can be a singleton (example: "Megagon Labs"), 
                                        #or it can be a sequency of an entity and its properties
                                        #example("Megagon Labs" || "parent company" or "Megagon Labs" || "parent company" || "location").
                                        #for the sake of simplicity, we will consider that "||" is the separator.
                                        #in both cases, the properties are not part of the filename.
    entityFileName = ''
    if len(entityInListFormat)>0:
        entityFileName = entityInListFormat[0]
    return entityFileName
        


def getListOfFileNamesFromDirectory(directory_path):
  """
  Returns a list of file names in the specified directory.

  Args:
    directory_path: The path to the directory.

  Returns:
    A list of strings, where each string is a file name.
    Returns an empty list if the directory does not exist or is empty.
  """
  try:
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return file_names
  except FileNotFoundError:
    return []

def addTriplesFromExtractor(pathToInputFilesDirectory,sourceOfInformation,currentDate,currentIteration,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources):    
    confidence = '0.1'
    listOfInputFileNames =  getListOfFileNamesFromDirectory(pathToInputFilesDirectory)
    provenanceInfo = 'instances file||'
    for inputFileName in listOfInputFileNames:
        inputFileNameFullPath = pathToInputFilesDirectory + '/' +inputFileName
        provenanceInfo = 'instances file||'+inputFileNameFullPath
        MyLogger().log('\n\n\n*******************************\nIngesting File '+str(inputFileNameFullPath))
        ingestFromTriples(inputFileNameFullPath, knowledgeBaseID, currentIteration, sourceOfInformation, currentDate, confidence, provenanceInfo, knowledgeBaseIDPath,fullyTrustedSources)


def addSeedInstances(triplesFileName,currentDate,currentIteration,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources):
    #triplesFileName = 'data/initializationTriples.tsv'
    sourceOfInformation = 'seedInstances'
    confidence = '1'
    provenanceInfo = 'seed instances file||'+str(triplesFileName)
    provenance = ''
    MyLogger().log('Calling "ingestFromTriples(triplesFileName, knowledgeBaseID, iterationNumber, sourceOfInformation, currentDate, confidence, provenanceInfo)"', level="INFO")
    ingestFromTriples(triplesFileName, knowledgeBaseID, currentIteration, sourceOfInformation, currentDate, confidence, provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)


def getValue(entity, relation, knowledgeBaseID,knowledgeBaseIDPath):
    valueGotten = None
    if entity != '':
        literalString = getFileNameFromEntity(entity).replace('\t',''.replace('\n','')) # splits the entity string based on || and return the initial subject of the entity
        canonicalString = canonicalizeString(literalString)#literalString.lower().replace(' ','_').replace('/','_')
        jsonFileName = createPhysicalFile(canonicalString[:200],knowledgeBaseIDPath + knowledgeBaseID.lower() + '/kb/')
        if jsonFileName == 'ERROR':
            MyLogger().log('Entity is empty.', level='ERROR')
            return valueGotten
        #jsonFileName = createPhysicalFile(canonicalString[:200],'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
        if os.path.isfile(jsonFileName):
            with open(jsonFileName, 'r') as propertiesJsonFile:
                jsonDict = json.load(propertiesJsonFile)
        else:
            return None
        if relation == 'theoSlot':
            return jsonDict
        tempDict = jsonDict.copy()
        for element in entity.split('||')[1:]:
            if tempDict.get(element) is None:
                return None
            if isinstance(tempDict[element],dict):
                tempDict = tempDict[element]
            else:
                return None
        if tempDict.get(relation) is None:
            return None
        valueGotten = tempDict[relation] 
    else:
        MyLogger().log('Entity is empty.', level='ERROR')
    return valueGotten


def getValeuByIteration():
    pass

def getValueBySource():
    pass

def getValueByConfidence():
    pass

def getFullPathSpecializationsValue(entity, knowledgeBaseID,knowledgeBaseIDPath):
    listOfSpecializations = None
    aditionalSpecializations = []
    if entity != '':
        listOfProcessedSpecializations = []
        listOfSpecializations = getValue(entity,'specializations',knowledgeBaseID,knowledgeBaseIDPath) 
        if listOfSpecializations != None:
            controlIndex = 0
            listOfSpecializations = list(listOfSpecializations.keys())
            while controlIndex < len(listOfSpecializations): #I'm using a while loop because the for loop might become instable when modifying the list that we are iterating over.
                targetSpecialization = listOfSpecializations[controlIndex]
                if targetSpecialization not in listOfProcessedSpecializations:
                    listOfProcessedSpecializations.append(targetSpecialization)
                    aditionalSpecializations = getValue(targetSpecialization,'specializations',knowledgeBaseID,knowledgeBaseIDPath) 
                    if aditionalSpecializations != None:
                        aditionalSpecializations = list(aditionalSpecializations.keys())
                        if aditionalSpecializations != []:
                            listOfSpecializations.extend(aditionalSpecializations)
                controlIndex += 1          
    else:
        MyLogger().log('Entity is empty.', level='ERROR')
    return listOfSpecializations



def putValue(entity, relation, valueToBePut, knowledgeBaseID, knowledgeBaseIDPath):
    if entity != '':
        literalString = getFileNameFromEntity(entity).replace('\t',''.replace('\n','')) # splits the entity string based on || and return the initial subject of the entity
        canonicalString = canonicalizeString(literalString)#literalString.lower().replace(' ','_').replace('/','_')
        jsonFileName = createPhysicalFile(canonicalString[:200],knowledgeBaseIDPath + knowledgeBaseID.lower() + '/kb/')
        if jsonFileName == 'ERROR':
            MyLogger().log('Entity is empty.', level='ERROR')
            return None
        #jsonFileName = createPhysicalFile(canonicalString[:200],'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
        entityInListFormat = entity.split('||')
        entityInListFormat.append(relation)
        #valueToBeAdded = canonicalizeString(valueToBeAdded).replace('_',' ')
        jsonDict = {}
        if os.path.isfile(jsonFileName):
            with open(jsonFileName, 'r') as propertiesJsonFile:
                jsonDict = json.load(propertiesJsonFile)
        putValueString = 'jsonDict'
        for element in entity.split('||')[1:]: #follow the full path to the final element in the entity
            if eval(putValueString).get(element) is None:
                putValueString = putValueString + '["' + element + '"]'
                execString = putValueString + ' = {}'
                try:
                    exec(execString)
                except:
                    return False
            else:
                putValueString = putValueString + '["' + element + '"]'

        #put the relation
        execString = putValueString + '["' + str(relation) + '"] = {}'
        putValueString = putValueString + '["' + str(relation) + '"]'
        try:
            exec(execString)
        except:
            return False
        execString = putValueString + '["' + str(valueToBePut) + '"] = {}'
        try:
            exec(execString)
        except:
            return False
        
        with open(jsonFileName, 'w') as jsonFile:
            json.dump(jsonDict,jsonFile, indent=4)
        return True
    else:
        MyLogger().log('Entity is empty.', level='ERROR')
        return False

def addPlusOneToTheLocalFrequency(relationSubject,targettargetIteration,targetSOurce,knowledgeBaseID,knowledgeBaseIDPath):
    #add +1 in the count of how many times this piece of knowledge was extracted
    currentFrequency = getValue(str(relationSubject)+'||iteration','totalFrequency',knowledgeBaseID,knowledgeBaseIDPath)
    if currentFrequency != None and currentFrequency != {}:
        currentFrequency = next(iter(currentFrequency))
        if type(int(currentFrequency)) == int:
            updatedFrequency = str(int(currentFrequency) + 1)
    else:
        updatedFrequency = '1'
    putValue(relationSubject,'totalFrequency',updatedFrequency,knowledgeBaseID,knowledgeBaseIDPath)


def addPlusOneToTheGeneralizations(relationSubject,targetFrequency,knowledgeBaseID,knowledgeBaseIDPath):
    #add +1 in the count of how many times this piece of knowledge was extracted
    updatedFrequency = 0
    currentFrequency = getValue(relationSubject,targetFrequency,knowledgeBaseID,knowledgeBaseIDPath)
    if currentFrequency != None and currentFrequency != {}:
        currentFrequency = next(iter(currentFrequency))
        if type(int(currentFrequency)) == int:
            updatedFrequency = str(int(currentFrequency) + 1)
    else:
        updatedFrequency = '1'
    putValue(relationSubject,targetFrequency,updatedFrequency,knowledgeBaseID,knowledgeBaseIDPath)
    return updatedFrequency

def tripleElementsAreNotEmpty(subject, predicate, object):
    if subject != '' and predicate != '' and object != '':
        return True
    else:
        return False


def createListOfConfidenceScores(givenConfidence,givenFrequency):
    confidenceScoresList = [givenConfidence] * int(givenFrequency)
    return confidenceScoresList

# Overall Confidence Update - Method 1: Noisy-OR
def noisyOr(confidences):
    return 1 - np.prod([1 - p for p in confidences])

# Overall Confidence Update - Method 2: Conservative Noisy-OR
def conservativeNoisyOr(confidences, penaltyTerm=0.75):
    extendedListOfCOnfidences = []
#    if confidences[0] == '1':
#        return (1,1)
#    else:
    for element in confidences:
        extendedElement = createListOfConfidenceScores(element[0],element[1])
        extendedListOfCOnfidences.extend(extendedElement)
    return 1 - np.prod([1 - (float(p)*penaltyTerm) for p in extendedListOfCOnfidences])

# Overall Confidence Update -  Method 3: Log-Odds Averaging
def logOddsAverage(confidences):
    logits = [logit(p) for p in confidences]
    avg_logit = np.mean(logits)
    return expit(avg_logit)

# Overall Confidence Update -  Method 4: Weighted Average (equal weights)
def weightedAverage(confidences, weights=None):
    if weights is None:
        weights = [1/len(confidences)] * len(confidences)
    return np.dot(confidences, weights)

def getConceptsAtIteration(targetEntity,targetIteration,knowledgeBaseID,knowledgeBaseIDPath):
    """
    This function returns a list with all concepts present in the kb in a given iteration
    """
    listOfSpecializations = None
    aditionalSpecializations = []
    listOfConceptsAtIteration = []
    if targetEntity != '' and targetEntity != None:
        listOfProcessedSpecializations = []
        listOfSpecializations = getValue(targetEntity,'specializations',knowledgeBaseID,knowledgeBaseIDPath) 
        if listOfSpecializations != None:
            controlIndex = 0
            listOfSpecializations = list(listOfSpecializations.keys())
            while controlIndex < len(listOfSpecializations): #I'm using a while loop because the for loop might become instable when modifying the list that we are iterating over.
                targetSpecialization = listOfSpecializations[controlIndex]
                if targetSpecialization not in listOfProcessedSpecializations:
                    listOfProcessedSpecializations.append(targetSpecialization)
                    targetSpecializationIteration = getValue(targetSpecialization,'specializations',knowledgeBaseID,knowledgeBaseIDPath) 
                    aditionalSpecializations = getValue(targetSpecialization,'specializations',knowledgeBaseID,knowledgeBaseIDPath) 
                    if aditionalSpecializations != None:
                        aditionalSpecializations = list(aditionalSpecializations.keys())
                        if aditionalSpecializations != []:
                            listOfSpecializations.extend(aditionalSpecializations)
                controlIndex += 1          
    else:
        MyLogger().log('Entity is empty.', level='ERROR')
    return listOfSpecializations

def getNumberOfConceptsAtIteration(targetIteration,knowledgeBaseID,knowledgeBaseIDPath):
    """
    This function returns the number of concepts present in the kb in a given iteration
    """
    pass

def findAllConfidenceScores(entity): 
    stackOfElements = list(entity.items()) 
    visitedElements = set()
    confidenceList = [] 
    while stackOfElements: 
        elementKey, elementValue = stackOfElements.pop() 
        if elementValue != {}:
            targetElementValue = next(iter(elementValue))
            if targetElementValue != 'confidence': 
                if elementKey not in visitedElements: 
                    stackOfElements.extend(elementValue.items()) 
            else: 
                confidenceList.append(next(iter(elementValue['confidence']))) 
        visitedElements.add(elementKey)
    return confidenceList

def findAllConfidenceScores_noLoop(entity,fullyTrustedSources): 
    stackOfElements = list(entity.items()) 
    visitedElements = set()
    confidenceList = [] 
    while stackOfElements: 
        elementKey, elementValue = stackOfElements.pop() 
        if elementValue != {}:
            targetElementValue = next(iter(elementValue))
            if targetElementValue == 'source':
                currentSource = next(iter(elementValue['source']))
                if currentSource in fullyTrustedSources:
                    MyLogger().log('The concept was added by a trusted source: '+str(currentSource) + '. Thus, the confidence is set to 1.')
                    return '1'
            if targetElementValue != 'confidence': 
                stackOfElements.extend(elementValue.items()) 
            else: 
                currentConfidence = next(iter(elementValue['confidence']))
                currentFrequency = next(iter(elementValue['confidence'][currentConfidence]['frequency']))
                confidenceList.append((currentConfidence,currentFrequency)) 
                MyLogger().log('Local confidence found: '+str(currentConfidence)+'. Frequecy: '+str(currentFrequency))
        visitedElements.add(elementKey)
    if confidenceList == []:
        MyLogger().log('No local confidence found!')
    overallConfidence = conservativeNoisyOr(confidenceList)
    return overallConfidence

def findAllMutuallyExclusiveInstances(relationSubject,relationPredicate,relationObject,knowledgeBaseID,knowledgeBaseIDPath):
    MyLogger().log('Searching for instances of mutually exclusive found for the triple: '+ str(relationSubject)+ '||'+str(relationPredicate)+'||'+str(relationObject))
    numberOfMutualExclusionInfractions = 0
    if relationPredicate == 'generalizations':
        #get the generalization of the parent concept
        parentsGeneralizations = getValue(relationObject,'generalizations',knowledgeBaseID,knowledgeBaseIDPath)
        if parentsGeneralizations:
            #get the subconcepts that are mutually exclusive with the relationObject and test whether the relationSubject is an instance of them
            mutuallyExclusiveRelationName = str(parentsGeneralizations)+'_is mutually exclusive with_'+str(parentsGeneralizations)
            mutualExclusiveConcepts = getValue(str(relationObject),mutuallyExclusiveRelationName,knowledgeBaseID,knowledgeBaseIDPath)
            if mutualExclusiveConcepts: 
                numberOfMutualExclusionInfractions = 0
                for mutualExclusiveConcept in mutualExclusiveConcepts:
                    listOfInstances = list(getValue(mutualExclusiveConcept,'specializations',knowledgeBaseID,knowledgeBaseIDPath).keys())
                    if relationSubject in listOfInstances:
                        numberOfMutualExclusionInfractions += 1
    else: 
        if relationPredicate == 'specializations':
            #get the subconcepts that are mutually exclusive with the relationObject and test whether the relationSubject is an instance of them
            generalizationConcepts = getValue(relationSubject,'generalizations',knowledgeBaseID,knowledgeBaseIDPath)
            if generalizationConcepts:
                for currentGeneralizationConcept in generalizationConcepts:
                    if currentGeneralizationConcept != 'concepts to be learned in the current kb':
                        mutuallyExclusiveRelationName = str(generalizationConcepts)+'_is mutually exclusive with_'+str(generalizationConcepts)
                        mutualExclusiveConcepts = getValue(str(relationSubject),mutuallyExclusiveRelationName,knowledgeBaseID,knowledgeBaseIDPath)
                        if mutualExclusiveConcepts: 
                            numberOfMutualExclusionInfractions = 0
                            for mutualExclusiveConcept in mutualExclusiveConcepts:
                                listOfInstances = list(getValue(mutualExclusiveConcept,'specializations',knowledgeBaseID,knowledgeBaseIDPath).keys())
                                if relationSubject in listOfInstances:
                                    numberOfMutualExclusionInfractions += 1
        else:
            if len(relationPredicate.split('_')) == 3:
                #get the type of the subject of the relation and the type of the object of the relation, then check whether relationSubject and relationObject are instances of them
                #for the subject of the relation:
                relationSubjectType = relationPredicate.split('_')[0]
                mutuallyExclusiveRelationName = str(relationSubjectType)+'_is mutually exclusive with_'+str(relationSubjectType)
                mutualExclusiveConcepts = getValue(str(relationSubjectType),mutuallyExclusiveRelationName,knowledgeBaseID,knowledgeBaseIDPath)
                if mutualExclusiveConcepts:
                    numberOfMutualExclusionInfractions = 0
                    for mutualExclusiveConcept in mutualExclusiveConcepts:
                        listOfInstances = list(getValue(mutualExclusiveConcept,'specializations',knowledgeBaseID,knowledgeBaseIDPath).keys())
                        if relationSubject in listOfInstances:
                            numberOfMutualExclusionInfractions += 1
                
                #for the object of the relation:
                relationObjectType = relationPredicate.split('_')[2]
                mutuallyExclusiveRelationName = str(relationObjectType)+'_is mutually exclusive with_'+str(relationObjectType)
                mutualExclusiveConcepts = getValue(str(relationObjectType),mutuallyExclusiveRelationName,knowledgeBaseID,knowledgeBaseIDPath)
                if mutualExclusiveConcepts:
                    for mutualExclusiveConcept in mutualExclusiveConcepts:
                        listOfInstances = list(getValue(mutualExclusiveConcept,'specializations',knowledgeBaseID,knowledgeBaseIDPath).keys())
                        if relationObject in listOfInstances:
                            numberOfMutualExclusionInfractions += 1
                MyLogger().log(str(numberOfMutualExclusionInfractions)+' instances of mutually exclusive found for the triple: '+ str(relationSubject)+ '||'+str(relationPredicate)+'||'+str(relationObject))
    return numberOfMutualExclusionInfractions

def overallConfidenceAupdate(relationSubject,relationPredicate,relationObject,tripletString,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources):
        try:
            if relationSubject!='concepts to be learned in the current kb' and relationObject != 'concepts to be learned in the current kb':
                MyLogger().log('Updating the overall confidence on the triplet: ' + tripletString)
                MyLogger().log('Getting the independent confidence scores')
                targetPath = getValue(relationSubject+'||'+relationPredicate,relationObject,knowledgeBaseID,knowledgeBaseIDPath)
                #confidenceList = findAllConfidenceScores(targetPath)
                overallConfidence = findAllConfidenceScores_noLoop(targetPath,fullyTrustedSources)
                mutuallyExclusiveOccurences = findAllMutuallyExclusiveInstances(relationSubject,relationPredicate,relationObject,knowledgeBaseID,knowledgeBaseIDPath)
                if mutuallyExclusiveOccurences:
                    MyLogger().log('Confidence score will be penalized because '+ str(mutuallyExclusiveOccurences)+' instances of mutually exclusive found for the triple: '+ tripletString)
                    overallConfidenceWithMutuallyExclusionPenalization = str(float(overallConfidence) / (float(mutuallyExclusiveOccurences)+1))
                else:
                    MyLogger().log('No instances of mutually exclusive found for the triple: '+ tripletString)
                    overallConfidenceWithMutuallyExclusionPenalization = str(float(overallConfidence))
                #overallConfidence = conservativeNoisyOr(confidenceList)
                MyLogger().log('Updating the overall confidence of the triple: '+ tripletString)
                putValue(relationSubject+'||'+relationPredicate+'||'+str(relationObject),'overall confidence',str(overallConfidenceWithMutuallyExclusionPenalization), knowledgeBaseID, knowledgeBaseIDPath)
                return overallConfidenceWithMutuallyExclusionPenalization
            else:
                return str(1)
        except Exception as e:
        # This catches any other type of exception
            MyLogger().log('An unexpected error occurred: ' + str(e))
            return None 

def overallConfidenceUpdate_old(relationSubject,relationPredicate,relationObject,tripletString,knowledgeBaseID,knowledgeBaseIDPath):
        try:
            MyLogger().log('Updating the overall confidence on the triplet: ' + tripletString)
            MyLogger().log('Getting the independent confidence scores')
            targetIterations = getValue(relationSubject+'||'+relationPredicate+'||'+relationObject,'iteration',knowledgeBaseID,knowledgeBaseIDPath).keys()
            processedSources = []
            overallConfidence = 0
            confidenceList = []
            for targetIteration in targetIterations:
                targetSources = getValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(targetIteration),'source',knowledgeBaseID,knowledgeBaseIDPath).keys()
                for targetSource in targetSources:
                    if targetSource == 'seedInstances':
                        MyLogger().log('This is a seed instance, so the confidence must be 1.')
                        putValue(relationSubject+'||'+relationPredicate+'||'+str(relationObject),'overall confidence','1', knowledgeBaseID, knowledgeBaseIDPath)
                        return '1'
                    processedSources.append(targetSource)
                    targetDates = getValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(targetIteration)+'||source||'+str(targetSource),'date',knowledgeBaseID,knowledgeBaseIDPath).keys()
                    for targetDate in targetDates:
                        targetConfidences = getValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(targetIteration)+'||source||'+str(targetSource)+'||date||'+str(targetDate),'confidence',knowledgeBaseID,knowledgeBaseIDPath).keys()
                        for targetConfidence in targetConfidences:
                            targetFrequency = next(iter(getValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(targetIteration)+'||source||'+str(targetSource)+'||date||'+str(targetDate)+'||confidence||'+str(targetConfidence),'frequency',knowledgeBaseID,knowledgeBaseIDPath).keys()))
                            for currentFrequency in range(int(targetFrequency)):
                                confidenceList.append(targetConfidence)
            overallConfidence = conservativeNoisyOr(confidenceList)
            MyLogger().log('Updating the overall confidence of the triple: '+ tripletString)
            putValue(relationSubject+'||'+relationPredicate+'||'+str(relationObject),'overall confidence',str(overallConfidence), knowledgeBaseID, knowledgeBaseIDPath)
            return overallConfidence
        except Exception as e:
        # This catches any other type of exception
            MyLogger().log('An unexpected error occurred: ' + str(e))
            return None 

def updateFrequencies(tripletString,knowledgeBaseID,knowledgeBaseIDPath):
    MyLogger().log('adding +1 in the count of how many times this triplet was extracted in this iteration, by this source, at this date:\ntriplet: ' + tripletString)
    targetFrequency = 'frequency'
    updatedFrequency = addPlusOneToTheGeneralizations(tripletString,targetFrequency,knowledgeBaseID,knowledgeBaseIDPath)
    #add +1 in the count of how many times this piece of knowledge was extracted
    targetFrequency = 'total frequency'
    tripletStringList = tripletString.split('||')
    relationSubject = tripletStringList[0]
    relationPredicate = tripletStringList[1]
    relationObject = tripletStringList[2]
    MyLogger().log('adding +1 in the total count of how many times the triplet was extracted: ' + relationSubject+'||'+relationPredicate+'||'+relationObject)
    updatedOverallFrequency = addPlusOneToTheGeneralizations(relationSubject+'||'+relationPredicate+'||'+relationObject,targetFrequency,knowledgeBaseID,knowledgeBaseIDPath)
    return updatedFrequency,updatedOverallFrequency


def getRelationInverseFromLLM(relationPredicate, sourceOfInformation):
    try:
        corePredicate = relationPredicate.split('_')[1]
        if corePredicate == 'is mutually exclusive with':
            return relationPredicate
        else:
            inverseFindingPrompt = 'You are an expert and seasoned taxonomist. \n\nProvide the name of the inverse relation of the given relation. Output only the name of the inverse relation. Do not include any additional text, punctuation, or explanation. \n\nExamples:  \n* the inverse of degrees_helps developing_skill is skills_is developed by_degree\n* the inverse of person_works for_companies is companies_employs_person\n* the inverse of author_writes_book is book_is_written by_author\n* the inverse of city_is capital of_country is country_has capital_city\n* the inverse of teacher_teaches_course is course_is taught by_teacher\n\nRelation: '+str(relationPredicate)#'You are an expert and seasoned taxonomist. \n\nprovide a name for the inverse of the original relation '+str(relationPredicate)+'.\n\nprovide only the name of the inverse relation and no other token.'
            MyLogger().log('prompting '+str(sourceOfInformation)+' using the following prompt: '+inverseFindingPrompt)
            inverseRelationName = probeOpenAI(inverseFindingPrompt)#call_gpt5_mini(inverseFindingPrompt)
            confidence = '0.6'
            MyLogger().log(str(sourceOfInformation) + 'response: '+ str(inverseRelationName))
            return inverseRelationName
    except Exception as e:
        # This catches any other type of exception
            MyLogger().log('An unexpected error occurred: ' + str(e))

def addValueFullPath(relationSubject,relationPredicate,relationObject,loggerMessage,provenanceInfo,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources):
    MyLogger().log(loggerMessage)
    tripletString = str(relationSubject) + '||' + str(relationPredicate) + '||' + str(relationObject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||'+ provenanceInfo+'||date||'+str(currentDate) 
    #addValue(tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence), knowledgeBaseID, knowledgeBaseIDPath)
    addValue(tripletString,'confidence',str(confidence), knowledgeBaseID, knowledgeBaseIDPath)
    #adding +1 in the count of how many times this triplet was extracted in this iteration, by this source
    updatedFrequency,updatedOverallFrequency = updateFrequencies(tripletString+'||confidence||'+str(confidence),knowledgeBaseID,knowledgeBaseIDPath)
    #update the overall confidence
    overallConfidence = overallConfidenceAupdate(relationSubject,relationPredicate,relationObject,tripletString,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
    #update The Iteration Content
    fullContent = tripletString+'||confidence||'+str(confidence)+'||frequency||'+ str(updatedFrequency) +'||total frequency||'+ str(updatedOverallFrequency)+'||overall confidence||'+str(overallConfidence)
    addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)    

def addValueInverseAndTypes(relationSubject,relationPredicate,relationObject,knowledgeBaseID,currentIteration,sourceOfInformation,currentDate, confidence, provenanceInfo, knowledgeBaseIDPath,fullyTrustedSources):
    #check if the predicate name is well-formed
    if tripleElementsAreNotEmpty(relationSubject, relationPredicate, relationObject) and len(relationPredicate.split('_')) == 3 or relationPredicate == 'generalizations' or relationPredicate == 'specializations':
        loggerMessage = 'adding the value, source, iteration, date and confidence of the generalization for the original triplet: '+ str(relationSubject) + '||' + str(relationPredicate) + '||' + str(relationObject)
        fullProvenance = provenanceInfo+'||this is the original triplet.'
        addValueFullPath(relationSubject,relationPredicate,relationObject,loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence, knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)

        #add the inverse
        tempRelationPredicate = relationPredicate.split('_')
        if len(tempRelationPredicate) == 1:
            if tempRelationPredicate[0] == 'generalizations':
                inverseRelationPredicate = 'specializations'
            elif tempRelationPredicate[0] == 'specializations':
                inverseRelationPredicate = 'generalizations'
            loggerMessage = 'Adding the value, source, iteration and date of the inverse of the original triplet: '+ str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
            fullProvenance = str(provenanceInfo)+'||inverse of original triplet||'+str(relationObject)+'_'+str(inverseRelationPredicate)+'_'+str(relationSubject)
            addValueFullPath(relationObject,inverseRelationPredicate,relationSubject,loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
            
        elif len(tempRelationPredicate) == 3:
            inverseRelationPredicate = getValue(relationPredicate,'has inverse',knowledgeBaseID,knowledgeBaseIDPath)
            if inverseRelationPredicate == None:
                inverseRelationPredicate = getRelationInverseFromLLM(relationPredicate,sourceOfInformation)    
                fullInverseRelationPredicate = str(inverseRelationPredicate)#tempRelationPredicate[2]+'_'+str(inverseRelationPredicate)+'_'+tempRelationPredicate[0]     
                addValue(relationPredicate,'has inverse',fullInverseRelationPredicate,knowledgeBaseID,knowledgeBaseIDPath)  
            else:
                fullInverseRelationPredicate = str(list(inverseRelationPredicate.keys())[0])
            #inverseRelationPredicate = 'inverse of_'+str(tempRelationPredicate)
            #MyLogger().log('adding the inverse triple: '+ str(relationObject) + '||' + str(inverseRelationPredicate) + '||' + str(relationSubject))
  
            #add the source, iteration and date of the inverse
            loggerMessage = 'Adding the value, source, iteration and date of the inverse from the original relation: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
            fullProvenance = provenanceInfo+'||inverse of original triplet||'+relationObject+'_'+fullInverseRelationPredicate+'_'+relationSubject
            addValueFullPath(relationObject,fullInverseRelationPredicate,relationSubject,loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                        
            if len(tempRelationPredicate)>1:
                subjectType = tempRelationPredicate[0]
                #add the generalizations for the subject type. For generalizations there is no need to define the type (as it is the type defiinition already). TODO: for specialization we are not defining, but we might want to do it in the future.
                #add the subject type, source, iteration and date
                loggerMessage = 'Adding the generalization, source, iteration, date and confidence for the subject of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||generalization of original triplet subject||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(relationSubject,'generalizations',str(tempRelationPredicate[0]),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                #add the source, iteration and date of the specialization of the subject type
                loggerMessage = 'Adding the specialization, source, iteration, date and confidence for the generalization of the subject of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||specialization of original triplet subject generalization||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(str(tempRelationPredicate[0]),'specializations',relationSubject,loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
 
                #add the generalizations for the object type
                #add the source, iteration and date of the generalization of the subject type
                loggerMessage = 'Adding the generalization, source, iteration, date and confidence for the of the object of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||generalization of original triplet object||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(relationObject,'generalizations',str(tempRelationPredicate[2]),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                #add the source, iteration and date of the specialization of the object generalization
                loggerMessage = 'Adding the specialization, source, iteration, date and confidence for the generalization of the object of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||specialization of original triplet object generalization||'+relationSubject+'_'+relationPredicate+'_'+relationObject
                addValueFullPath(str(tempRelationPredicate[2]),'specializations',relationObject,loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
 
                #add the generalizations of the relation/predicate and its source, iteration and date of the generalization
                loggerMessage = 'Adding the generalization, source, iteration, date and confidence for the relation of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||generalization of the relation of the original triplet||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(str(relationPredicate),'generalizations','relations',loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                
                #add the specializations of the relation/predicate, and its source, iteration and date of the specialization
                loggerMessage = 'Adding the specialization, source, iteration, date and confidence for the relation of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||specialization of the relation of the original triplet||'+relationSubject+'_'+relationPredicate+'_'+relationObject
                addValueFullPath('relations','specializations',str(relationPredicate),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
            
                #add the instance pair for the relation/predicate, and its source, iteration and date of the specialization
                instancePair = (relationSubject, relationObject)
                loggerMessage = 'Adding the instance pair, source, iteration, date and confidence for the relation of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||instance pair of the relation of the original triplet||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(relationPredicate,'relation_has instance pairs_instance pairs',str(instancePair),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                #now do it for the inverse of the relation
                #add the source, iteration and date of the specialization
                loggerMessage = 'Adding the specialization, source, iteration, date and confidence for the inverse of the relation of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||inverse of the relation of the original triplet||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath('relations','specializations',str(fullInverseRelationPredicate),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
                #add the instance pair for the inverse of the relation/predicate as an instance of the relation "relation_has instance pairs_instance pairs"
                inverseInstancePair = (relationObject, relationSubject)
                loggerMessage = 'Adding the instance pair, source, iteration, date and confidence for the inverse of the relation of the triplet: '+str(relationSubject)+'||'+str(relationPredicate)+'||'+str(relationObject)
                fullProvenance = provenanceInfo+'||instance pair of the inverse of the relation of the original triplet||'+str(relationSubject)+'_'+str(relationPredicate)+'_'+str(relationObject)
                addValueFullPath(fullInverseRelationPredicate,'relation_has instance pairs_instance pairs',str(inverseInstancePair),loggerMessage,fullProvenance,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath,fullyTrustedSources)
    else:
        MyLogger().log('The triple is not well-formed. Ignoring it', level="ERROR")

def getRelationsFromTriples(triplesFolderName):
    seedFilesList = getListOfFileNamesFromDirectory(triplesFolderName)
    setOfRelationsToBeLearned = set({})
    for seedFileName in seedFilesList:
        try: 
            with open(str(triplesFolderName)+'/'+str(seedFileName), 'r') as triplesFile:
                for line in triplesFile:
                    splittedLine = line.replace('\n','').split('\t')
                    if len(splittedLine)==3:
                        if str(splittedLine[1])+'\tgeneralizations\trelations to be learned in the current kb' not in setOfRelationsToBeLearned:
                            setOfRelationsToBeLearned.add(str(splittedLine[1])+'\tgeneralizations\trelations to be learned in the current kb')
                            setOfRelationsToBeLearned.add(str(splittedLine[1])+'\tgeneralizations\trelations')#+str(seedFileName))
        except FileNotFoundError:
            print(str(seedFileName)+ ' not found.', level="ERROR")
        with open('relationsInitialization.txt','w') as relationsInitFile:
            relationsInitFile.write('\n'.join(sorted(setOfRelationsToBeLearned)))
        print('File relationsInitialization.txt created.')

def addValueInverseAndTypes_SlowVersion_Deprecated(relationSubject,relationPredicate,relationObject,knowledgeBaseID,currentIteration,sourceOfInformation,currentDate, confidence, promptTemplate, promptInstance, knowledgeBaseIDPath):
    #DEPRECATED!!!
    #check if the predicate name is well-formed
    if tripleElementsAreNotEmpty(relationSubject, relationPredicate, relationObject) and len(relationPredicate.split('_')) == 3 or relationPredicate == 'generalizations' or relationPredicate == 'specializations':
        tripletString =  str(relationSubject) + '||' + str(relationPredicate) + '||' + str(relationObject) 
        sourceOfInformation = sourceOfInformation + '_Original Triplet: ' + tripletString
        #add the value
        MyLogger().log('adding triple: '+ str(relationSubject) + '\t' + str(relationPredicate) + '\t' + str(relationObject))
        valueAdded = addValue(relationSubject,relationPredicate,relationObject,knowledgeBaseID, knowledgeBaseIDPath)
        if valueAdded == None:
            MyLogger().log('Triple is malformed and was not added: '+str(relationSubject) + '\t' + str(relationPredicate) + '\t'+ str(relationObject))
            return None
        MyLogger().log('Triple Added: ' + str(relationSubject) + '||' + str(relationPredicate) + '||' + str(relationObject))
        #add source, iteration, date and confidence of the generalization
        MyLogger().log('adding the source, iteration, date and confidence of the generalization for: '+ str(relationSubject) + '||' + str(relationPredicate) + '||' + str(relationObject))
        valueAdded = addValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence), knowledgeBaseID, knowledgeBaseIDPath)
        if valueAdded == None:
            MyLogger().log('Triple is malformed and was not added: '+str(relationSubject) + '||' + str(relationPredicate) + '||'+ str(relationObject))
            return None
        #adding +1 in the count of how many times this triplet was extracted in this iteration, by this source
        updateFrequencies(relationSubject,relationPredicate,relationObject,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
        if len(promptTemplate) > 1:
            #add prompt
            MyLogger().log('Adding the prompt used by the LLM to extract the triple: '+tripletString)
            addValue(relationSubject+'||'+relationPredicate+'||'+relationObject+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)
        #update the overall confidence
        overallConfidence = overallConfidenceAupdate(relationSubject,relationPredicate,relationObject,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
        #update The Iteration Content
        fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
        addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)

        #add the inverse
        tempRelationPredicate = relationPredicate.split('_')
        inverseRelationPredicate = 'inverse of_'+str(tempRelationPredicate)
        MyLogger().log('adding the inverse triple: '+ str(relationObject) + '||' + str(inverseRelationPredicate) + '||' + str(relationSubject))
        if len(tempRelationPredicate) == 1: #it means that the relation is "generalizations" or "specializations"
            if tempRelationPredicate[0] == 'generalizations':
                inverseRelationPredicate = 'specializations'
            elif tempRelationPredicate[0] == 'specializations':
                inverseRelationPredicate = 'generalizations'
            valueAdded=addValue(relationObject,inverseRelationPredicate,relationSubject, knowledgeBaseID, knowledgeBaseIDPath)
            if valueAdded == None:
                MyLogger().log('Triple is malformed and was not added: '+str(relationSubject) + '\t' + str(relationPredicate) + '\t'+ str(relationObject))
                return None
            #print('Triple Added: ' + str(relationObject) + ', ' + str(inverseRelationPredicate) + ', ' + str(relationSubject))
            MyLogger().log('Triple Added: ' + str(relationObject) + ', ' + str(inverseRelationPredicate) + ', ' + str(relationSubject))
            #add the source, iteration and date of the inverse
            MyLogger().log('Adding the source, iteration and date of the inverse')
            addValue(relationObject+'||'+inverseRelationPredicate+'||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence), knowledgeBaseID, knowledgeBaseIDPath)
            #adding +1 in the count of how many times this triplet was extracted in this iteration, by this source
            updateFrequencies(relationObject,inverseRelationPredicate,relationSubject,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationObject) + '||' + str(inverseRelationPredicate) + '||' + str(relationSubject) 
            overallConfidence = overallConfidenceAupdate(relationObject,inverseRelationPredicate,relationSubject,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            
            if len(promptTemplate) > 1:
            #add prompt
                MyLogger().log('Adding the prompt used by the LLM')
                addValue(relationObject+'||'+inverseRelationPredicate+'||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)
        else: #it means that the relation is not "generalizations" or "specializations", thus we will also need to add the types of the subject and the object
            inverseRelationPredicate = tempRelationPredicate[2]+'_inverse of '+tempRelationPredicate[1]+'_'+tempRelationPredicate[0]
            MyLogger().log('adding triple: '+ str(relationObject) + '\t' + str(inverseRelationPredicate) + '\t' + str(relationSubject))
            addValue(relationObject,inverseRelationPredicate,relationSubject, knowledgeBaseID, knowledgeBaseIDPath)
            MyLogger().log('Triple Added: ' + str(relationObject) + ', ' + str(inverseRelationPredicate) + ', ' + str(relationSubject))
            #add the source, iteration and date of the inverse
            MyLogger().log('Adding the source, iteration and date of the inverse')
            addValue(relationObject+'||'+inverseRelationPredicate+'||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence), knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(relationObject,inverseRelationPredicate,relationSubject,currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationObject) + '||' + str(inverseRelationPredicate) + '||' + str(relationSubject) 
            overallConfidence = overallConfidenceAupdate(relationObject,inverseRelationPredicate,relationSubject,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)

            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM.')
                addValue(relationObject+'||'+inverseRelationPredicate+'||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the generalizations for the subject type
            if len(tempRelationPredicate)>1:
                subjectType = tempRelationPredicate[0]
            else:
                subjectType = 'everything' #TODO: for generalizations and specializations there is still a need to define a process to identify the type. For now it's been assumed 'everything' as a default type
            MyLogger().log('adding the subject type triple: '+ str(relationSubject) + '\tgeneralizations\t' + str(subjectType))
            addValue(relationSubject,'generalizations',subjectType,knowledgeBaseID, knowledgeBaseIDPath)
            MyLogger().log('Triple Added: ' + str(relationSubject) + ', ' + 'generalizations' + ', ' + str(tempRelationPredicate[0]))
            #add the source, iteration and date of the generalization
            MyLogger().log('Adding the source, iteration and date of the subject type')
            addValue(relationSubject+'||generalizations||'+str(tempRelationPredicate[0])+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times the subject type was extracted
            updateFrequencies(relationSubject,'generalizations',str(tempRelationPredicate[0]),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationSubject) + '||generalizations||' + str(subjectType) 
            overallConfidence = overallConfidenceAupdate(relationSubject,'generalizations',subjectType,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LMM')
                addValue(relationSubject+'||generalizations||'+str(tempRelationPredicate[0])+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the specializations for the subject type
            MyLogger().log('adding the triple for the specializations of the subject type: '+ str(tempRelationPredicate[0]) + '\tspecializations\t' + str(relationSubject))
            addValue(tempRelationPredicate[0],'specializations',relationSubject,knowledgeBaseID, knowledgeBaseIDPath)
            MyLogger().log('Triple Added: ' + str(tempRelationPredicate[0]) + ', ' + 'specializations' + ', ' + str(relationSubject))
            #add the source, iteration and date of the specialization of the subject type
            MyLogger().log('Adding the source, iteration and date of the specialization of the subject type')
            addValue(tempRelationPredicate[0]+'||specializations||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(tempRelationPredicate[0],'specializations',str(relationSubject),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(tempRelationPredicate[0]) + '||specializations||'+ str(relationSubject) 
            overallConfidence = overallConfidenceAupdate(tempRelationPredicate[0],'specializations',relationSubject,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)

            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM')
                addValue(tempRelationPredicate[0]+'||specializations||'+str(relationSubject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the generalizations for the object type
            MyLogger().log('adding triple for the generalizations for the object type: '+ str(relationObject) + '\tgeneralizations\t' + str(tempRelationPredicate[2]))
            addValue(relationObject,'generalizations',tempRelationPredicate[2],knowledgeBaseID, knowledgeBaseIDPath)
            #print('Triple Added: ' + str(relationObject) + ', ' + 'generalizations' + ', ' + str(tempRelationPredicate[2]))
            #add the source, iteration and date of the generalization of the subject type
            MyLogger().log('adding the source, iteration and date of the generalization of the subject type')
            addValue(relationObject+'||generalizations||'+str(tempRelationPredicate[2])+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(relationObject,'generalizations',str(tempRelationPredicate[2]),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationObject) + '||generalizations||'+ str(tempRelationPredicate[2]) 
            overallConfidence = overallConfidenceAupdate(relationObject,'generalizations',tempRelationPredicate[2],tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM')
                addValue(relationObject+'||generalizations||'+str(tempRelationPredicate[2])+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the specializations for the object type
            MyLogger().log('Adding the triple for the specializations of the object type: '+str(tempRelationPredicate[2]) + ', ' + 'specializations' + ', ' + str(relationObject))
            addValue(tempRelationPredicate[2],'specializations',relationObject,knowledgeBaseID, knowledgeBaseIDPath)
            #print('Triple Added: ' + str(tempRelationPredicate[2]) + ', ' + 'specializations' + ', ' + str(relationObject))
            #add the source, iteration and date of the specialization of the object type
            MyLogger().log('Adding the source, iteration and date of the specialization of the object type')
            addValue(tempRelationPredicate[2]+'||specializations||'+str(relationObject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(tempRelationPredicate[2],'specializations',str(relationObject),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(tempRelationPredicate[2]) + '||specializations||' + str(relationObject) 
            overallConfidence = overallConfidenceAupdate(tempRelationPredicate[2],'specializations',relationObject,tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)

            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM')
                addValue(tempRelationPredicate[2]+'||specializations||'+str(relationObject)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the generalizations of the relation/predicate
            MyLogger().log('Adding the triple for the generalizations of the relation/predicate: '+ str(relationPredicate) + ', ' + 'generalizations, relations')
            addValue(relationPredicate,'generalizations','relations',knowledgeBaseID, knowledgeBaseIDPath)
            #print('Triple Added: ' + str(relationPredicate) + ', ' + 'generalizations, relations')
            #add the source, iteration and date of the generalization
            MyLogger().log('Adding the source, iteration and date of the generalization of the relation/predicate')
            addValue(relationPredicate+'||generalizations||relations||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(relationPredicate,'generalizations','relations',currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationPredicate) + '||generalizations||relations' 
            overallConfidence = overallConfidenceAupdate(relationPredicate,'generalizations','relations',tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)

            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM')
                addValue(relationPredicate+'||generalizations||relations||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the specializations of the relation/predicate as an instance of relation
            MyLogger().log('Adding the triple for the specializations of the relation/predicate as an instance of relation: relations, specializations, ' + str(relationPredicate))
            addValue('relations','specializations',relationPredicate,knowledgeBaseID, knowledgeBaseIDPath) #TODO: VERY IMPORTANT: if we addValue for this pair of instances, the canonicalization process will transform "[argument1, argument2]" into "argument1 argument2", thus, for now we will skip the addValue and will add it in the next command with the source and confidence information.
                                                                                    #this error only occurs for the value to be added to the "relation_has instance pairs_instance pairs" slot
            #print('Triple Added: relations, specializations, ' + str(relationPredicate))
            #add the source, iteration and date of the specialization
            MyLogger().log('Adding the source, iteration and date of the specialization')
            addValue('relations||specializations||'+str(relationPredicate)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies('relations','specializations',str(relationPredicate),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = 'relations||specializations||' +str(relationPredicate) 
            overallConfidence = overallConfidenceAupdate('relations','specializations',str(relationPredicate),tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM.')
                addValue('relations||specializations||'+str(relationPredicate)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #add the instance pair for the relation/predicate as an instance of the relation "relation_has instance pairs_instance pairs"
            MyLogger().log('add the instance pair for the relation/predicate as an instance of the relation "relation_has instance pairs_instance pairs".')
            tuplePair = (relationSubject, relationObject)
            addValue(relationPredicate,'relation_has instance pairs_instance pairs',str(tuplePair),knowledgeBaseID, knowledgeBaseIDPath)
            MyLogger().log('Triple Added: '+str(relationPredicate)+', relation_has instance pairs_instance pairs, '+str(tuplePair))
            #add the source, iteration and date of the specialization
            MyLogger().log('Adding the source, iteration and the date of the specialization of the instance of the relation "relation_has instance pairs_instance pairs"')
            addValue(str(relationPredicate)+'||relation_has instance pairs_instance pairs||'+str(tuplePair)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(str(relationPredicate),'relation_has instance pairs_instance pairs',str(tuplePair),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(relationPredicate) + '||relation_has instance pairs_instance pairs||'+str(tuplePair) 
            overallConfidence = overallConfidenceAupdate(str(relationPredicate),'relation_has instance pairs_instance pairs',str(tuplePair),tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM.')
                addValue(str(relationPredicate)+'||relation_has instance pairs_instance pairs||'+str(tuplePair)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)

            #now do it for the inverse of the relation
            #add the specializations for the inverse of the relation/predicate as an instance of relation
            MyLogger().log('Adding the specializations for the inverse of the relation/predicate as an instance of relation: relations, specializations, ' + str(inverseRelationPredicate))
            addValue('relations','specializations',inverseRelationPredicate,knowledgeBaseID, knowledgeBaseIDPath) #TODO: VERY IMPORTANT: if we addValue for this pair of instances, the canonicalization process will transform "[argument1, argument2]" into "argument1 argument2", thus, for now we will skip the addValue and will add it in the next command with the source and confidence information.
                                                                                    #this error only occurs for the value to be added to the "relation_has instance pairs_instance pairs" slot
            #print('Triple Added: relations, specializations, ' + str(inverseRelationPredicate))
            #add the source, iteration and date of the specialization
            MyLogger().log('the source, iteration and date of the specialization of the inverse of the relation/predicate as an instance of relation "relation_has instance pairs_instance pair"')
            addValue('relations||specializations||'+str(inverseRelationPredicate)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(str('relations'),'specializations',str(inverseRelationPredicate),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = 'relations||specializations||'+ str(inverseRelationPredicate) 
            overallConfidence = overallConfidenceAupdate('relations','specializations',str(inverseRelationPredicate),tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM.')
                addValue('relations||specializations||'+str(inverseRelationPredicate)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID,knowledgeBaseIDPath)

            #add the instance pair for the inverse of the relation/predicate as an instance of the relation "relation_has instance pairs_instance pairs"
            listPair = (relationObject, relationSubject)
            MyLogger().log('Adding the pair of instances for the inverse of the relation/predicate as an instance of relation: '+str(inverseRelationPredicate)+', relation_has instance pairs_instance pairs, '+str(listPair))
            addValue(inverseRelationPredicate,'relation_has instance pairs_instance pairs',str(listPair),knowledgeBaseID, knowledgeBaseIDPath)
            #print('Triple Added: '+str(inverseRelationPredicate)+', relation_has instance pairs_instance pairs, '+str(listPair))
            #add the source, iteration and date of the specialization
            MyLogger().log('the source, iteration and date of the pair of instances of the inverse of the relation/predicate as an instance of relation "relation_has instance pairs_instance pair"')
            addValue(str(inverseRelationPredicate)+'||relation_has instance pairs_instance pairs||'+str(listPair)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate),'confidence',str(confidence),knowledgeBaseID, knowledgeBaseIDPath)
            #add +1 in the count of how many times this piece of knowledge was extracted
            updateFrequencies(str(inverseRelationPredicate),'relation_has instance pairs_instance pairs',str(listPair),currentIteration,sourceOfInformation,currentDate,confidence,knowledgeBaseID,knowledgeBaseIDPath)
            #update the overall confidence
            tripletString = str(inverseRelationPredicate) + '||relation_has instance pairs_instance pairs||'+ ', ' +str(listPair) 
            overallConfidence = overallConfidenceAupdate(str(inverseRelationPredicate),'relation_has instance pairs_instance pairs',str(listPair),tripletString,knowledgeBaseID,knowledgeBaseIDPath)
            #update The Iteration Content
            fullContent = tripletString+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||confidence||'+str(confidence)+'||overall confidence||'+str(overallConfidence)
            addValue(str(currentIteration)+'th_iteration', 'content',fullContent,knowledgeBaseID,knowledgeBaseIDPath)
            if len(promptTemplate) > 1:
                #add prompt
                MyLogger().log('Adding the prompt used by the LLM.')
                addValue(str(relationPredicate)+'||relation_has instance pairs_instance pairs||'+str(listPair)+'||iteration||'+str(currentIteration)+'||source||'+str(sourceOfInformation)+'||date||'+str(currentDate)+'||prompt template||',str(promptTemplate)+'prompt instance',str(promptInstance), knowledgeBaseID, knowledgeBaseIDPath)
    else:
        MyLogger().log('The triple is not well-formed. Ignoring it', level="ERROR")

def canonicalizeString(originalString):
    canonicalString = unicodedata.normalize('NFKC', originalString)[:200]
    canonicalString = re.sub(r'[^\w\s-]', '', canonicalString.lower())
    canonicalString = re.sub(r'[-\s]+', '_', canonicalString).strip('-_')
    return canonicalString.replace('\t','').replace('\n','')

def addValue(entity, relation, valueToBeAdded, knowledgeBaseID, knowledgeBasePath):
    if tripleElementsAreNotEmpty(entity,relation,valueToBeAdded):
        literalString = getFileNameFromEntity(entity).replace('\t',''.replace('\n','')) # splits the entity string based on || and return the initial subject of the entity
        canonicalString = canonicalizeString(literalString)#literalString.lower().replace(' ','_').replace('/','_')
        #jsonFileName = createPhysicalFile(canonicalString[:200],'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
        jsonFileName = createPhysicalFile(canonicalString[:200], knowledgeBasePath + knowledgeBaseID.lower() + '/kb/')
        if jsonFileName == 'ERROR':
            MyLogger().log('Entity is empty.', level='ERROR')
            return None
        entityInListFormat = entity.split('||')
        entityInListFormat.append(relation)
        #valueToBeAdded = canonicalizeString(valueToBeAdded).replace('_',' ')
        jsonDict = {}
        if os.path.isfile(jsonFileName):
            with open(jsonFileName, 'r') as propertiesJsonFile:
                jsonDict = json.load(propertiesJsonFile)
        tempDict = jsonDict.copy()
        addValueString = 'jsonDict'
            #add the canonical string
        if eval(addValueString).get('canonical string') is None: # if the canonical string relation is not present yet, then add it
            execString = addValueString + '["' +  str('canonical string') + '"] = {}'
            try:
                exec(execString)
            except:
                return False
        execString = addValueString + '["' +  str('canonical string') + '"]'
        if eval(execString).get(canonicalString) is None: # if the canonical string value is not present yet, then add it
            execString = execString + '["' +  str(canonicalString) + '"] = {}'
            try:
                exec(execString)
            except:
                return False
        #add the literal string
        if eval(addValueString).get('literal string') is None: # if the canonical string relation is not present yet, then add it
            execString = addValueString + '["' +  str('literal string') + '"] = {}'
            try:
                exec(execString)
            except:
                return False    
        execString = addValueString + '["' +  str('literal string') + '"]'
        if eval(execString).get(literalString) is None: # if the literal string value is not present yet, then add it
            execString = execString + '["' +  str(literalString) + '"] = {}'
            try:
                exec(execString)
            except:
                return False
        addValueString = 'jsonDict'
        listOfKeys = []
        for element in entity.split('||')[1:]: #follow the full path to the final element in the entity
            addValueString = addValueString + '["' + element + '"]'
            if tempDict.get(element) is None:
                execString = addValueString + ' = {}'
                try:
                    exec(execString)
                except:
                    return False
            else:
                tempDict = tempDict[element]

        #add the relation
        if eval(addValueString).get(relation) is None: #add the relation to be used to put the value to
            execString = addValueString + '["' + str(relation) + '"] = {}'
            try:
                exec(execString)
            except:
                return False
        addValueString = addValueString + '["' + str(relation) + '"]'
        if eval(addValueString).get(valueToBeAdded) is None: #finally add the new value
            execString = addValueString + '["' + str(valueToBeAdded) + '"] = {}'
            try:
                exec(execString)
            except:
                return False
        
        with open(jsonFileName, 'w') as jsonFile:
            json.dump(jsonDict,jsonFile, indent=4)
        return True
    else:
        with open(str(knowledgeBasePath + knowledgeBaseID.lower())+'/malformedTriplesNotIngested.txt', 'a') as malformedTriplesNotIngestedFile:
            malformedTriplesNotIngestedFile.write('\n')
        MyLogger().log('Skipping because the entity name is empty!',level="ERROR")


def getConceptsToBeLearned(knowledgeBaseID,knowledgeBaseIDPath):
    rawConceptsList = getValue('concepts to be learned in the current kb','specializations',knowledgeBaseID,knowledgeBaseIDPath)
    finalConceptList = []
    if rawConceptsList != None:
        for conceptTBeLearned in rawConceptsList:
            finalConceptList.append(conceptTBeLearned)
    return finalConceptList

def getRelationsToBeLearned(knowledgeBaseID,knowledgeBaseIDPath):
    rawRelationsList = getValue('relations to be learned in the current kb','specializations',knowledgeBaseID,knowledgeBaseIDPath)
    finalRelationtList = []
    if rawRelationsList != None:
        for relationToBeLearned in rawRelationsList:
            finalRelationtList.append(relationToBeLearned)
        return finalRelationtList
    else:
        return {}

def getExamplesOfConcept(conceptToBeLearned,knowledgeBaseID, knowledgeBaseIDPath):
    currentExamples = getValue(conceptToBeLearned,'specializations', knowledgeBaseID,knowledgeBaseIDPath)
    listOfSelectedExamples = []
    if currentExamples != None: 
        listOfExamples = list(currentExamples.keys())
        if len(listOfExamples)>10:
            listOfSelectedExamples = random.sample(listOfExamples, k=10)
        else:
            listOfSelectedExamples = listOfExamples
        MyLogger().log('List of instances selected out from '+str(len(listOfExamples))+' instances present in the KB: '+str(listOfSelectedExamples))

        for selectedEntity in listOfSelectedExamples:
            try:
                isASlot = getValue(selectedEntity,'is a slot',knowledgeBaseID,knowledgeBaseIDPath)
                if isASlot == 'theoSlot':
                    listOfSelectedExamples.remove(selectedEntity)
                    MyLogger().log('removing '+str(selectedEntity)+' from the list of examples, because it is a theoSlot.')
            except:
                pass
        #TODO:apply any filter/ranking/sorting
    return listOfSelectedExamples
    
def getSampleInstancesFromRelation(targetRelation,knowledgeBaseID,knowledgeBaseIDPath):
    MyLogger().log('getting the pair instances for the relation '+str(targetRelation)+'.')
    currentExamplesDict = getValue(targetRelation,'relation_has instance pairs_instance pairs', knowledgeBaseID,knowledgeBaseIDPath)
    listOfSelectedExamples = []
    finalList = []
    if currentExamplesDict != None: 
        listOfExamples = list(currentExamplesDict.keys())
        if len(listOfExamples)>10:
            listOfSelectedExamples = random.sample(listOfExamples, k=10)
        else:
            listOfSelectedExamples = listOfExamples
        MyLogger().log('List of pairs selected out of '+str(len(listOfExamples))+' pairs present in the KB: '+str(listOfSelectedExamples))
        for selectedEntity in listOfSelectedExamples:
            try:
                relationSubject = eval(selectedEntity)[0]
                relationObject = eval(selectedEntity)[1]
                isASlotSubject = getValue(relationSubject,'is a slot',knowledgeBaseID,knowledgeBaseIDPath)
                isASlotObject = getValue(relationObject,'is a slot',knowledgeBaseID,knowledgeBaseIDPath)
                if isASlotSubject == 'theoSlot' or isASlotObject == 'theoSlot':
                    listOfSelectedExamples.remove(selectedEntity)
                    MyLogger().log('removing '+str(selectedEntity)+' from the list of examples, because there is a theoSlot in one of the elements.')
                else:
                    finalList.append(eval(selectedEntity))
            except:
                pass
        #TODO:apply any filter/ranking/sorting
    return finalList



def getRelationInstancesFromLLMs(knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence, relatioPromptTemplate,relationToBePopulated,objectType,relationInstancesListPairs,relationName,knowledgeBaseIDPath,fullyTrustedSources):
    listOfLearnedIntancePairs = []
    relationPrompt = 'Filed prompt building from template: '+ relatioPromptTemplate
    if 'llama' in sourceOfInformation:
        pass
        #try:
        #    relationPrompt = eval(relatioPromptTemplate)
        #    relationInstances, modelName = 'probevLLM(relationPrompt)'#TODO: replace the call for an api model
        #    sourceOfInformation = modelName
        #    confidence = '0.6'
        #    #relations, modelName = probeVllmModelWithRelationsGenericPrompt(relationInitialListPairs,relationPromptTemplates,relationPromptList,relationName,subjectType,predicateType,objectType,numberOfExtractions,iterationNumber,knowledgeBaseID, spacyModel)
        #    print(relationInstances)
        #    listOfLearnedIntancePairs = eval(relationInstances[0].split('\n\n')[1])
        #except: 
        #    print('Error in prompt: ' + relatioPromptTemplate)
    else:
        if 'gpt' in sourceOfInformation:
            try:
                relationPrompt = eval(relatioPromptTemplate)
                MyLogger().log('prompting '+str(sourceOfInformation)+' using the following prompt: '+relationPrompt)
                relationInstances = probeOpenAI(relationPrompt)
                confidence = '0.6'
                listOfLearnedIntancePairs = eval(relationInstances)#eval(relations[0].split('\n\n')[1])
                MyLogger().log(str(sourceOfInformation) + 'response: '+ str(listOfLearnedIntancePairs))
            except: 
                MyLogger().log('Error in prompt: ' + relationPrompt)
    provenanceInfo = 'prompt template||'+str(relatioPromptTemplate)+'||prompt||'+str(relationPrompt)       
    if len(listOfLearnedIntancePairs) >1:
        for learnedPair in listOfLearnedIntancePairs:
            learnedTripletSubject = learnedPair[0]
            learnedTripletObject = learnedPair[1]
            addValueInverseAndTypes(learnedTripletSubject,relationToBePopulated,learnedTripletObject,knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)

def getConceptInstancesFromLLms(knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence, promptTemplates, conceptToBeLearned, subjectType,listOfSeedExamples,knowledgeBaseIDPath,fullyTrustedSources):
    extractedNames = []
    currentPrompt = ''
    if 'llama' in sourceOfInformation:
        pass
        """        
        try:
            currentPrompt = eval(promptTemplates)
            extractedTokens, modelName = probevLLM(currentPrompt)
            sourceOfInformation = modelName
            confidence = '0.6'
            extractedNames = eval(extractedTokens[0].split('\n\n')[1])
        except:
            print('Error in prompt: ' + promptTemplates) """
    else:
        if 'gpt' in sourceOfInformation:
            try:
                currentPrompt = eval(promptTemplates)
                extractedTokens = probeOpenAI(currentPrompt)
                #sourceOfInformation = modelName
                extractedNames = eval(extractedTokens)

                confidence = '0.6'
            except:
                MyLogger().log('Error in prompt: ' + promptTemplates)
    provenanceInfo = 'prompt template||'+str(promptTemplates[0])+'||prompt||'+str(currentPrompt)
    for extractedItem in extractedNames:
        tripletSubject = extractedItem
        tripletPredicate = 'generalizations'
        tripletObject = conceptToBeLearned
        addValueInverseAndTypes(tripletSubject,tripletPredicate,tripletObject,knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence,provenanceInfo,knowledgeBaseIDPath,fullyTrustedSources)  #addToJsonFile(tripletSubject,str(promptTemplate),prompt()



def getValue_old(entity, relation, knowledgeBaseID, logFileName):
    uncannonicalizedJsonFileName = getFileNameFromEntity(entity)
    jsonFileName = createPhysicalFile(uncannonicalizedJsonFileName.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    entityInListFormat = entity.split('||')
    with open(jsonFileName, 'r') as propertiesJsonFile:
        jsonDict = json.load(propertiesJsonFile)
    valueRetrievalString = 'jsonDict'#.get("' + uncannonicalizedJsonFileName +'", {})'
    for element in entity.split('||')[1:]:
        valueRetrievalString = valueRetrievalString + '.get("' + element + '", {})'
    valueRetrievalString = valueRetrievalString + '.get("' + relation + '")'
    MyLogger().log('Getting value of ' + valueRetrievalString, file=logFileName)
    valueGotten  = eval(valueRetrievalString) #jsonDict.get("outer", {}).get("inner")
    return valueGotten

def addValue_old(entity, relation, addValue, knowledgeBaseID):
    uncannonicalizedJsonFileName = getFileNameFromEntity(entity)
    jsonFileName = createPhysicalFile(uncannonicalizedJsonFileName.lower().replace(' ','_').replace('/','_'),'./data/kbs/' + knowledgeBaseID.lower() + '/kb/')
    entityInListFormat = entity.split('||')
    with open(jsonFileName, 'r') as propertiesJsonFile:
        jsonDict = json.load(propertiesJsonFile)
    valueRetrievalString = 'jsonDict'#.get("' + uncannonicalizedJsonFileName +'", {})'
    addValueString = 'jsonDict'
    for element in entity.split('||')[1:]:
        valueRetrievalString = valueRetrievalString + '.get("' + element + '", {})'
        addValueString = addValueString + '["' + element + '"]'
    valueRetrievalString = valueRetrievalString + '.get("' + relation + '")'
    MyLogger().log('Putting value of ' + addValueString)
    existingValue  = eval(valueRetrievalString) #jsonDict.get("outer", {}).get("inner")
    if existingValue == None: #TODO: actually, I need to check the whole path to identify whath piece is missing (that caused the None)
        addValueString = addValueString + ' = ' + addValue
    else:
        eval(addValueString)
    return addValue

def updateLastIterationInParametersJsonFile(knowledgeBaseID, updatedLastIteration,knowledgeBaseIDPath):
    propertiesJsonFileName = knowledgeBaseIDPath + str(knowledgeBaseID) + '/parameters.json'
    with open(propertiesJsonFileName, 'r') as parametersJsonFile:
        parametersJson = json.load(parametersJsonFile)
        parametersJson[0]['lastIteration'] = str(updatedLastIteration)
    with open(propertiesJsonFileName, 'w') as propertiesJsonFile:
        json.dump(parametersJson, propertiesJsonFile, indent = 4)
    return propertiesJsonFileName


def main(text):
    print("text: ", text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", type=str, default="postgres")
    parser.add_argument("--user", type=str, default="postgres")
    parser.add_argument("--password", type=str, default="postgres")
    parser.add_argument("--prompt_path", type=str, default="prompts/prompt.json")
    args = parser.parse_args()
    main(args)


"""
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text")

    args = parser.parse_args()

    main(args.text)
"""