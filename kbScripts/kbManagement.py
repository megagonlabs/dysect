from datetime import datetime
import itertools
import json
import os
import spacy
import re
import basicLib as bLib
import random

spacyModel = spacy.load("en_core_web_lg")

#sentenceBertModel = SentenceTransformer('all-MiniLM-L6-v2')

#todos will be added

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

def fetchRelationPromptsFromFile_vllm(promptsFileName):
    promptList = []
    with open(promptsFileName, 'r' ) as promptsFile: #reads a json file containing a list of prompt dictionaries
                                    #with keys: "prompt", "promptType" (can be "seedTemplate" or "learnedTemplate")
                                    #and "confidenceScore" (the confidence the model/system has that the prompt is
                                    # precise in extracting good candidates)
        promptDictList = json.load(promptsFile)
    for prompt in promptDictList:
        assembledPrompt = prompt['prompt'].replace('relationName',prompt['relationName'])
        assembledPrompt = assembledPrompt.replace('subjectType',prompt['subjectType'])
        assembledPrompt = assembledPrompt.replace('objectType',prompt['objectType'])
        if prompt['MASK_TYPE'] == 'objectType':
            promptList.append(((assembledPrompt),(prompt['MASK_PAIR'],'[MASK]')))
        else:
            if prompt['MASK_TYPE'] == 'subjectType':
                promptList.append(((assembledPrompt),('[MASK]',prompt['MASK_PAIR'])))
    return promptList

def getInitialPairsOfConceptSeeds(relationToBePopulated):
    initialList = []
    seedPairsFileName = str(relationToBePopulated)
    print(seedPairsFileName)
    initialList = readListOfEntitiesFromTxtFile(seedPairsFileName)
    return initialList

def getDictFromJsonFile(jsonFileName):
    if os.path.exists(jsonFileName):
        with open(jsonFileName, 'r') as jsonFile:
            dictFromJson = json.load(jsonFile)
        return dictFromJson
    else:
        return {}


def readListOfEntitiesFromTxtFile(txtInputFile):
    listOfEntities = []
    try:
        with open(txtInputFile, 'r') as inputFile:
            fileContent = inputFile.read()
            listOfEntities = eval(fileContent)
    except:
        return [] # if the file can't be read, just return the empty list       
    return listOfEntities


def getInitialConceptSeeds(concept):
    initialList = []
    seedsFileName = 'concepts/'+ str(concept) + '.txt'
    print(seedsFileName)
    initialList = readListOfEntitiesFromTxtFile(seedsFileName)
    return initialList

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

def fetchPromptsFromFile_vllm(promptsFileName):
    promptList = []
    with open(promptsFileName, 'r' ) as promptsFile: #reads a json file containing a list of prompt dictionaries
                                    #with keys: "prompt", "promptType" (can be "seedTemplate" or "learnedTemplate")
                                    #and "confidenceScore" (the confidence the model/system has that the prompt is
                                    # precise in extracting good candidates)
        promptDictList = json.load(promptsFile)
    for prompt in promptDictList:
        promptList.append(prompt['prompt'])
    return promptList

def fetchrelationPrompts_vLLMFromFile(prompts):
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

def probevLLM(promptForVllm):
    prompts = '[vLLMlib.tokenizer.apply_chat_template([{"role": "user", "content": promptForVllm}], tokenize=False), vLLMlib.tokenizer.apply_chat_template([{"role": "user", "content": promptForVllm}], tokenize=False)]'#TODO: replace with an api call
    #Just test the vLLM
    #vLLM.run('test_inputs_v0.jsonl', 'test_outputs_v1.jsonl','meta-llama/Llama-2-7b-chat-hf',0.,1.0,128,None,'model_input')
    responses = 'vLLMlib.llm.generate(prompts, sampling_params=vLLMlib.sampling_params)' #TODO:replace with an api call
    listOfGeneratedOutputs = []
    for llmresponse in responses:
        for llmoutput in llmresponse.outputs:
            print(llmoutput.text.strip())
            listOfGeneratedOutputs.append(llmoutput.text.strip())
    return listOfGeneratedOutputs, 'vLLMlib.llm.llm_engine.model_config.model'#TODO:replace with an api call

def seedsFromOnet(knowledgeBaseID, ingestionIteration, startOfCurrentIterationDate, confidence):
    #ingest skills from O*Net
    ingestionFileFileName = 'data/onet/db_29_1_text/Skills.txt'
    ingestionConcept = 'skills'
    bLib.ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence)

    #ingest abilities from O*Net
    ingestionFileFileName = 'data/onet/db_29_1_text/Abilities.txt'
    ingestionConcept = 'abilities'
    bLib.ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence)

    #ingest knowledge from O*Net
    ingestionFileFileName = 'data/onet/db_29_1_text/Knowledge.txt'
    ingestionConcept = 'knowledge'
    bLib.ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence)

    #ingest Work Activities from O*Net
    ingestionFileFileName = 'data/onet/db_29_1_text/Work Activities.txt'
    ingestionConcept = 'work activities'
    bLib.ingestFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence)
    
    #ingest Occupation Descriptions from O*Net
    ingestionFileFileName = 'data/onet/db_29_1_text/Occupation Data.txt'
    ingestionConcept = 'descriptions'
    bLib.ingestOccupationDescriptionsFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID,confidence)

    #ingest Tasks from O*Net
    #ingestionFileFileName = 'data/onet/db_29_1_text/Task Statements.txt'
    #ingestionConcept = 'tasks'
    #bLib.ingestSkillsFileFromOnet(ingestionConcept,ingestionFileFileName,ingestionIteration,startOfCurrentIterationDate, knowledgeBaseID)


def ingestOccupationDescriptionsFromOnet(concept,ingestionFileFileName, iteration,ingestionDate, knowledgeBaseID):

    testValue=bLib.getValue('chemistry||generalizations||knowledge','totalCount','starting-dec-19th-2024')
    print(testValue)
    testValue = bLib.putValue('chemistry||generalizations||knowledge||testing','testando','2nNewValue','starting-dec-19th-2024')
    triplesFileName = 'data/onet/title_requires knowledge on_knowledge.tsv'
    ingestionFileFileName = 'data/onet/onetKnowledge.tsv'
    bLib.ingestTriplesBatchFromFile(triplesFileName, '0', 'oNet', 'starting-dec-19th-2024')
    os.system('export CUDA_VISIBLE_DEVICES=4')
    # spacyModel = spacy.load("en_core_web_lg")
    spacyModel = None
    prompts = json.load(open('prompts/prompt.json', "r"))   
    relationPrompts = json.load(open("prompts/prompt_relation_vllm.json", "r"))
    print("loading model")
    concept = 'companies'
    relations = bLib.getRelationsFromInitialDirectory('relations/')
    iterationNumber = 0
    initialDayOfIteration = str(datetime.now().isoformat()).split('T')[0]

def probeVllmModelWithRelationsGenericPrompt(relationInstancesListPairs,relationPromptTemplates,relationPromptList,relationName,subjectType,predicateType,objectType,numberOfExtractions,iterationNumber,knowledgeBaseID, spacyModel):
    extractedNames = []
    totalNumberOfPairs = len(relationInstancesListPairs)
    currentPair = 0
    currentPair += 1
    for promptTemplate in relationPromptTemplates:
        print('Iteration: ' + str(iterationNumber)+ '_' + knowledgeBaseID + '. Processing pair ' + str(currentPair) + ' out of ' + str(totalNumberOfPairs) + ' for relation: ' + str(relationName) + '. ')
        print(promptTemplate)
        relationPromptList.append(promptTemplate)
        currentPrompt = promptTemplate
        prompt = eval(currentPrompt[0])
        print(prompt)
        try:
            extractedTokens, modelName = 'probevLLM(prompt)' #TODO: replace this by an api call
        except:
            print('Error in prompt: ' + prompt)
            continue
        extractedNames = extractedTokens[0].split(' = ')[1].split('```')
        
        for extractedItem in eval(extractedNames[0]):
            tripletSubject = extractedItem[0]
            tripletObject = extractedItem[1]
            cannonicalSubject = tripletSubject.lower()
            bLib.addToJsonFile(tripletSubject,str(promptTemplate),prompt,tripletSubject,relationName,tripletObject,iterationNumber,knowledgeBaseID, modelName)
        
            
    return extractedNames

def getRelationsToBeLearned(knowledgeBaseID):
    rawRelationsList = bLib.getValue('relations','specializations',knowledgeBaseID)
    finalRelationtList = []
    for relationToBeLearned in rawRelationsList:
        finalRelationtList.append(relationToBeLearned)
    return finalRelationtList

def getConceptsToBeLearned(knowledgeBaseID):
    rawConceptsList = bLib.getValue('concepts to be learned in current kb','specializations',knowledgeBaseID)
    finalConceptList = []
    if rawConceptsList != None:
        for conceptTBeLearned in rawConceptsList:
            finalConceptList.append(conceptTBeLearned)
    return finalConceptList

def getExamplesOfConcept(conceptToBeLearned,knowledgeBaseID):
    currentExamples = bLib.getValue(conceptToBeLearned,'specializations', knowledgeBaseID)
    listOfSelectedExamples = []
    if currentExamples != None: 
        listOfExamples = list(currentExamples.keys())
        if len(listOfExamples)>15:
            listOfSelectedExamples = random.sample(listOfExamples, k=10)
            for selectedEntity in listOfSelectedExamples:
                try:
                    isASlot = bLib.getValue(selectedEntity,'is a slot',knowledgeBaseID)
                    if isASlot == 'theoSlot':
                        listOfSelectedExamples.remove(selectedEntity)
                except:
                    pass
        #TODO:apply any filter/ranking/sorting
    return listOfSelectedExamples
    
def getSampleInstancesFromRelation(targetRelation,knowledgeBaseID):
    currentExamplesDict = bLib.getValue(targetRelation,'relation_has instance pairs_instance pairs', knowledgeBaseID)
    listOfSelectedExamples = []
    finalList = []
    if currentExamplesDict != None: 
        listOfExamples = list(currentExamplesDict.keys())
        if len(listOfExamples)>15:
            listOfSelectedExamples = random.sample(listOfExamples, k=10)
        else:
            listOfSelectedExamples = listOfExamples
        for selectedEntity in listOfSelectedExamples:
            try:
                isASlot = bLib.getValue(selectedEntity,'is a slot',knowledgeBaseID)
                if isASlot == 'theoSlot':
                    listOfSelectedExamples.remove(selectedEntity)
                else:
                    finalList.append(eval(selectedEntity))
            except:
                pass
        #TODO:apply any filter/ranking/sorting
    return finalList



def getRelationInstancesFromLLMs(knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence, relatioPromptTemplate,relationToBePopulated,objectType,relationInstancesListPairs,relationName):
    listOfLearnedIntancePairs = []
    if 'llama' in sourceOfInformation:
        try:
            relationPrompt = eval(relatioPromptTemplate)
            relationInstances, modelName = 'probevLLM(relationPrompt)'#TODO: replace the call for an api model
            sourceOfInformation = modelName
            confidence = '0.6'
            #relations, modelName = probeVllmModelWithRelationsGenericPrompt(relationInitialListPairs,relationPromptTemplates,relationPromptList,relationName,subjectType,predicateType,objectType,numberOfExtractions,iterationNumber,knowledgeBaseID, spacyModel)
            print(relationInstances)
            listOfLearnedIntancePairs = eval(relationInstances[0].split('\n\n')[1])
        except: 
            print('Error in prompt: ' + relatioPromptTemplate)
    else:
        if 'gpt' in sourceOfInformation:
            try:
                relationPrompt = eval(relatioPromptTemplate)
                relationInstances = bLib.probeOpenAI(relationPrompt)
                confidence = '0.6'
                #relations, modelName = probeVllmModelWithRelationsGenericPrompt(relationInitialListPairs,relationPromptTemplates,relationPromptList,relationName,subjectType,predicateType,objectType,numberOfExtractions,iterationNumber,knowledgeBaseID, spacyModel)
                print(relationInstances)
                listOfLearnedIntancePairs = eval(relationInstances)#eval(relations[0].split('\n\n')[1])
            except: 
                print('Error in prompt: ' + relatioPromptTemplate)
    if len(listOfLearnedIntancePairs) >1:
        for learnedPair in listOfLearnedIntancePairs:
            learnedTripletSubject = learnedPair[0]
            learnedTripletObject = learnedPair[1]
            bLib.addValueInverseAndTypes(learnedTripletSubject,relationToBePopulated,learnedTripletObject,knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence,relatioPromptTemplate, relationPrompt)

def getConceptInstancesFromLLms(knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence, promptTemplates, conceptToBeLearned, subjectType,listOfSeedExamples):
    extractedNames = []
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
                extractedTokens = bLib.probeOpenAI(currentPrompt)
                #sourceOfInformation = modelName
                extractedNames = eval(extractedTokens)

                confidence = '0.6'
            except:
                print('Error in prompt: ' + promptTemplates)

    for extractedItem in extractedNames:
        tripletSubject = extractedItem
        tripletPredicate = 'generalizations'
        tripletObject = conceptToBeLearned
        bLib.addValueInverseAndTypes(tripletSubject,tripletPredicate,tripletObject,knowledgeBaseID,iterationNumber,sourceOfInformation,currentDate,confidence, promptTemplates[0],currentPrompt)                #bLib.addToJsonFile(tripletSubject,str(promptTemplate),prompt()

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


def main():
    #os.system('export CUDA_VISIBLE_DEVICES=5')    
    currentIteration = 0
    startOfCurrentIterationDate = str(datetime.now().isoformat()).split('T')[0]
    dateForKBNameAndLogFile =  str(datetime.now().isoformat())
    rootParametersFileName = 'parameters.json'
    if os.path.isfile(rootParametersFileName):
        with open(rootParametersFileName, "r", encoding="utf-8") as parametersFile:
            parametersData = json.load(parametersFile)
        if parametersData[0]['knowledgeBaseID'] == '':
            knowledgeBaseID = str(dateForKBNameAndLogFile.lower())
            parametersData[0]['knowledgeBaseID'] = knowledgeBaseID
        else:
            knowledgeBaseID = parametersData[0]['knowledgeBaseID']
    else:
        knowledgeBaseID = str(dateForKBNameAndLogFile.lower())
    knowledgeBaseIDDir = '/path/to/data/kbs/'+ str(knowledgeBaseID) + '/'
    if not os.path.isdir(knowledgeBaseIDDir):
        os.makedirs(knowledgeBaseIDDir)
    os.system("cp -r 'prompts' " + knowledgeBaseIDDir)
    bLib.MyLogger().configure(str(knowledgeBaseIDDir)+ 'run_'+ str(dateForKBNameAndLogFile) + '.log')
    bLib.MyLogger().log('Log File: '+str(str(knowledgeBaseIDDir)+ 'run_'+ str(startOfCurrentIterationDate) + '.log'))
    bLib.MyLogger().log('Starting run_'+ str(startOfCurrentIterationDate), level="INFO")
    bLib.MyLogger().log('If ./parameters.log knowledgeBaseID was "", then a new knowldgeBaseID was created based on the current date and time. Otherwise, the knowledgeBaseID present in the original ./parametes.json was used.' )
    

    parametersFileName = str(knowledgeBaseIDDir) + str(rootParametersFileName)
    if not os.path.isfile(parametersFileName):
        os.system('cp '+ str(rootParametersFileName) + ' ' + parametersFileName)
        with open(parametersFileName, "r", encoding="utf-8") as parametersFile:
            parametersData = json.load(parametersFile)
        parametersData[0]['knowledgeBaseID'] = knowledgeBaseID
        currentIteration = parametersData[0]['lastIteration']
        with open(parametersFileName, "w", encoding="utf-8") as parametersFile:
            json.dump(parametersData, parametersFile)
    kbParameters = bLib.loadParameters('parameters.json',knowledgeBaseID, currentIteration)    
    maxIterations = kbParameters[0]['maxIterations']
    relationPromptsFileName_vllm = '/path/to/data/kbs/'+str(knowledgeBaseID)+'/prompts/prompt_relation_vllm.json'
    bLib.knowledgeIntegrator(knowledgeBaseID, currentIteration, 'everything', '0.8', 'frequency', startOfCurrentIterationDate)
    #add seed instances
    triplesFileName = 'data/initializationTriples.tsv'
    sourceOfInformation = 'seedInstances'
    currentDate = startOfCurrentIterationDate 
    confidence = '1'
    promoptTemplate = ''
    currentPrompt = ''
    bLib.MyLogger().log('Calling "bLib.ingestFromTriples(triplesFileName, knowledgeBaseID, iterationNumber, sourceOfInformation, currentDate, confidence, promoptTemplate, currentPrompt)"', level="INFO")
    bLib.ingestFromTriples(triplesFileName, knowledgeBaseID, currentIteration, sourceOfInformation, currentDate, confidence, promoptTemplate, currentPrompt)
    triplesFileName = 'data/sample-data.txt'
    sourceOfInformation = 'Extractor_data_science-base-v4_4o'#'Extractor_data_science-skills-v4_4o'
    currentDate = startOfCurrentIterationDate 
    confidence = '0.5'
    listOfInputFileNames =  getListOfFileNamesFromDirectory('/path/to/data/extractions/data_science/base/v4_4o/adjusted')
    for inputFileName in listOfInputFileNames:
        inputFileNameFullPath = '/path/to/data/extractions/data_science/base/v4_4o/adjusted/' + inputFileName
        bLib.MyLogger().log('\n\n\n*******************************\nIngesting File '+str(inputFileNameFullPath))
        bLib.ingestFromTriples(inputFileNameFullPath, knowledgeBaseID, currentIteration, sourceOfInformation, currentDate, confidence, promoptTemplate, currentPrompt)
    #add seeds from O*Net
    confidence = '0.99'
    #seedsFromOnet(knowledgeBaseID, ingestionIteration, startOfCurrentIterationDate, confidence)

    while currentIteration < 0: #maxIterations:
        #set the KB parameters
        #the parameters should be available in a "parameters.json" file in the root directory of the project
        #in the 1st iteration, the properties.json file will be copied to the specific data/kbs/knowledgeBaseID directory, so that 
        #it is easy to know the configurations used to achieve the results stored in that specific KB
        #In the 1st call of the "loadParameters" we pass "none" as the knowledgeBaseID in order to create a new KB. In case we want to run starting from a previously created KB, we should paas the KB ID instead of "none"
        lastIteration = currentIteration
        
        print(kbParameters)
        #knowledgeBaseID = kbParameters[0]['knowledgeBaseID']
        frequencyThreshold = kbParameters[0]['frequencyThreshold']
        maxIterations = kbParameters[0]['maxIterations']
        numberOfExtractions = kbParameters[0]['numberOfExtractions'] #number of entities to be fetched from the model for each prompt
        promotionCriteria = kbParameters[0]['promotionCriteria']
        lastIteration = kbParameters[0]['lastIteration']
        ###########################################
        
        startOfCurrentIterationDate = str(datetime.now().isoformat()).split('T')[0]
        #TODO: run the knowledgeIntegrator to update the KB and get ready for the next iteration
        kbPath = '/path/to/data/kbs/'+str(knowledgeBaseID)+'/kb/'
        bLib.knowledgeIntegrator(knowledgeBaseID,currentIteration,'everything', frequencyThreshold, promotionCriteria,startOfCurrentIterationDate)
        currentIteration += 1
        print('starting iteration ' + str(currentIteration))
        #initialList = getInitialConceptSeeds(concept) 
        listOfConceptsToBeLearned = getConceptsToBeLearned(knowledgeBaseID) 
        print(listOfConceptsToBeLearned)
        entityListPairs = list(itertools.combinations(listOfConceptsToBeLearned, 2))
        promptsFileName = 'data/kbs/'+str(knowledgeBaseID)+'/prompts/prompt_vllm.json'
        promptTemplates = fetchPromptsFromFile_vllm(promptsFileName)
        promptList = []
        confidence = '0.6'
        for conceptToBeLearned in listOfConceptsToBeLearned:
            print('\n--------------------Starting to retrieve instances for the concept '+str(conceptToBeLearned)+'\n--------------------')
            subjectType = conceptToBeLearned
            listOfSeedExamples = getExamplesOfConcept(conceptToBeLearned,knowledgeBaseID)
            #extract content from llama
            sourceOfInformation = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            getConceptInstancesFromLLms(knowledgeBaseID, currentIteration,sourceOfInformation,currentDate,confidence,promptTemplates[0],conceptToBeLearned, subjectType,listOfSeedExamples)
            
            #extract content from gpt
            sourceOfInformation = "gpt-4o-mini"
            getConceptInstancesFromLLms(knowledgeBaseID, currentIteration,sourceOfInformation,currentDate,confidence,promptTemplates[0],conceptToBeLearned, subjectType,listOfSeedExamples)
        listOfRelations = getRelationsToBeLearned(knowledgeBaseID)#getRelationsForConcept(concept)
        relationPromptTemplates = fetchRelationPromptsFromFile_vllm(relationPromptsFileName_vllm)
        for relationToBePopulated in listOfRelations:
            print('processing relation '+str(relationToBePopulated))
            if len(relationToBePopulated.split('_'))>1: #TODO: for now we do not populate relations that do not have the subject and object types
                relationInstancesListPairs = getSampleInstancesFromRelation(relationToBePopulated, knowledgeBaseID)#getInitialPairsOfConceptSeeds('relations/'+relationToBePopulated + '.txt')
                relationName = relationToBePopulated.split('.txt')[0]
                subjectType = relationName.split('_')[0]
                predicateType = relationName.split('_')[1]
                objectType = relationName.split('_')[2]
                #relationListPairs = list(itertools.combinations(relationInitialListPairs, 2))
                if predicateType[:10] != 'inverse of': # TODO: for now, we do not add values for the inverse relations. We might want to do it in the future.
                    #extract content from llama
                    sourceOfInformation = "meta-llama/Meta-Llama-3.1-8B-Instruct"
                    getRelationInstancesFromLLMs(knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,relationPromptTemplates[0][0],relationToBePopulated,objectType,relationInstancesListPairs,relationName)

                    #extract content from gpt
                    sourceOfInformation = "gpt-4o-mini"
                    getRelationInstancesFromLLMs(knowledgeBaseID,currentIteration,sourceOfInformation,currentDate,confidence,relationPromptTemplates[0][0],relationToBePopulated, objectType, relationInstancesListPairs,relationName)


        propertiesFileName = bLib.updateLastIterationInParametersJsonFile(knowledgeBaseID,currentIteration) # consolidate the last iteration to the local copy of the parameters.json file


if __name__ == '__main__':
    main()
