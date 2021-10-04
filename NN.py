import argparse
import keras
import numpy
import matplotlib.pyplot as plt
import time
import datetime
import pandas
import json
import h5py
import os
import itertools
import random
import sys
import tensorflow as tf
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

parser = argparse.ArgumentParser(description='Generates NeuralNetwork')
parser.add_argument('-design',required=True,help='Path to Designfolder')
parser.add_argument('-data',required=True,help='Path to CSV Inputdata')
parser.add_argument('-encoding',required=True,help='Encoding file path')
parser.add_argument('-evaluate',required=False,default=False,action='store_true',help='Evaluate the model. Default = False')
parser.add_argument('-evaluate_only',required=False,default=False,action='store_true',help='Only evaluate the model. Default = False')
parser.add_argument('-shuffle',required=False,default=False,action='store_true',help='Shuffle the dataframe. Default = False')
parser.add_argument('-no_save',required=False,default=False,action='store_true',help='Do not saves the generated model. Default = False')
parser.add_argument('-reload',required=False,default=False,action='store_true',help='Reloads a generated model. Default = False')
parser.add_argument('-single',required=False,default=False,action='store_true',help='Use single model to predict multistate. Default = False')
parser.add_argument('-state',required=False,default=0,type=int,help='Specific State to clalc')
parser.add_argument('-operators',required=False,default=['+'],type=list,help='Operators to use as fitness function. Default = +')
parser.add_argument('-pop',required=False,default=120,type=int,help='Number of new sequences. Default = 120')
parser.add_argument('-gen',required=False,default=1,type=int,help='Number of next generation. Default = 1')
parser.add_argument('-random_seq',required=False,default=2000000,type=int,help='Number of random sequences. Default = 2.000.000')
parser.add_argument('-random_seq_split',required=False,default=20,type=int,help='Split the random sequences generation, MEM save. Default = 20')
args=parser.parse_args()

Encodepath = os.path.abspath(args.encoding)
Datapath = os.path.abspath(args.data)
os.chdir(os.path.abspath(args.design))


if args.evaluate_only:
    args.evaluate = True

def gen_data(Encoding_Dic):
    #Generates the data from checkpoint files
    with open(Datapath,'r') as file_h:
        content = file_h.readlines()

    #How many states
    line = content[0]
    line = line.split(',')
    Total_States = len(line) - 2 # -1 for Sequence and -1 for Sum Score
    States_to_use = Total_States
    if args.single:
        States_to_use = 1

    Data = []
    for line in content:
        line = line.split(',')
        Tempsequence = line[0]
        Tempsequence = [x for x in Tempsequence]
        Tempsequence = encode(Encoding_Dic,Tempsequence)
        if States_to_use == 1:
            Tempsequence.append(float(line[1]))
        else:
            for score in range(States_to_use):
                Tempsequence.append(float(line[2+score]))
        Data.append(Tempsequence)
    return(Data,Total_States,States_to_use)

def encode(Encoding_Dic,Tempsequence):
    Encoded_Seq = []
    for AA in Tempsequence:
        Encoded_Seq.extend(Encoding_Dic[AA])
    return(Encoded_Seq)

def reload_model(state,gen):
    print('Reloading Model %d from generation %d' %(state+1,gen))
    model_json = './NN_Log/model_%i_%i.json' % (state+1,gen)
    weights = './NN_Log/model_%i_%i.h5' % (state+1,gen)
    with open(model_json, 'r') as json_file:
        model_json = json_file.read()
    model = keras.models.model_from_json(model_json)
    model._make_predict_function()
    model.load_weights(weights)
    return(model)

def save_model(model,state,gen):
    model_json = './NN_Log/model_%i_%i.json' % (state+1,gen)
    weights = './NN_Log/model_%i_%i.h5' % (state+1,gen)
    json_model = model.to_json()
    with open(model_json,'w') as json_file:
        json_file.write(json_model)
    model.save_weights(weights)

def Get_Scores():
    with open(Datapath,'r') as score_file:
        content = score_file.readlines()
    Score_Dic = {}
    add = 1E-8
    for line in content:
        line = line.split(',')
        Sequence = line[0]
        Score = float(line[1])
        while Score in Score_Dic.keys():
            Score += add #Keys need to be unique

        Score_Dic[Score] = Sequence
    return(Score_Dic)

def random_seq(Seeds,AA_pool):
    Sequences = []
    print('Building random')
    Total = int(args.random_seq / args.random_seq_split) # reduce mem used -> run multiple times
    if 20**len(Seeds[0]) < Total:
        Total = 20**len(Seed)
    for i in range(int(Total)):
        Seed = Seeds[random.randint(0,9)]
        Seed = list(Seed)
        Positions = list(range(len(Seed)))
        random.shuffle(Positions)
        Positions = Positions[0:random.randint(1,len(Seed))]
        Tempsequence = Seed[:]
        for Position in Positions:
            Tempsequence[Position] = AA_pool[random.randint(0,len(AA_pool)-1)]
        Sequences.append(''.join(Tempsequence))
    return(Sequences)

def main():
    def Neural_Network(i):
        Y = States_Y[i] #Prediction targets
        start_time = time.time() #Time
        model = keras.Sequential()
        Neurons_1 = 10 + int(X.shape[1]/2)
        model.add(keras.layers.Dense(Neurons_1,input_shape=(X.shape[1],),activation='tanh',kernel_initializer='RandomNormal',use_bias=0))
        Neurons_2 = 5 + int(X.shape[1]/4)
        model.add(keras.layers.Dense(Neurons_2,activation='tanh',kernel_initializer='RandomNormal',use_bias=0))
        model.add(keras.layers.Dense(1,activation='linear',kernel_initializer='RandomNormal',use_bias=0))
        epochen = 1
        sgd = keras.optimizers.SGD(lr = 0.1,momentum = 0.8, decay=0.1/(epochen*X.shape[0]),nesterov=False)
        model.compile(loss='mse',optimizer='sgd')
        history = model.fit(X,Y,epochs = epochen,validation_split=0.01,batch_size=1)
        learning_time = numpy.round(time.time() - start_time)
        print('Learning took: %s h:m:s' % str(datetime.timedelta(seconds=learning_time)))
        return(model)

    if not os.path.isdir('NN_Log'):
        os.mkdir('NN_Log')

    #Generate Encoding_Dic
    with open(Encodepath,'r') as file_h:
        content = file_h.readlines()
    content = [x.strip() for x in content]
    Encoding_Dic = {}
    for line in content:
        line = line.split(',')
        Encoding_Dic[line[0]] = []
        for item in line[1:]:
            Encoding_Dic[line[0]].append(float(item)) #float is required here!
    Data,Total_States,States_to_use = gen_data(Encoding_Dic)
    DF = pandas.DataFrame(Data).astype(float)

    #shuffle Dataset
    if args.shuffle:
        DF = DF.sample(frac=1).reset_index(drop=True)
    #Split state specific
    X = DF.iloc[:,0:-States_to_use]
    States_Y = []
    for i in range(States_to_use):
        States_Y.append(DF.iloc[:,-(States_to_use - i)])

    pop_size = args.pop
    Gen = args.gen
    with open('./NN_Log/Log_%i.txt' % (Gen),'w') as log_file:
        log_file.write('Data successfully generated\nPop_size: %i\nPredicting generation: %i\n' % (pop_size,Gen+1))

    #Only split training if Evaluating
    if args.evaluate:
        print('Spliting data in trainingset and testset')
        X_Test = X[::3]
        X = X.drop(list(X_Test.index.values))
        States_Y_Test = []
        for i, Y in enumerate(States_Y):
            States_Y_Test.append(Y[::3])
            Y = Y.drop(list(States_Y_Test[-1].index.values))
            States_Y[i] = Y

    #Normalizing
    print('Normalizing')
    X_Max = max(X.max())
    X = X.apply(lambda a: a / float(X_Max))
    if args.evaluate:
        X_Test = X_Test.apply(lambda a: a / float(X_Max))
    Y_norms = []
    for i in range(States_to_use):
        Y = States_Y[i]
        Mean_Y = numpy.mean(Y)
        Std_Y = numpy.std(Y)
        if args.evaluate:
            States_Y_Test[i] = (States_Y_Test[i] - Mean_Y) / Std_Y
        States_Y[i] = (Y - Mean_Y) / Std_Y
        Y_norms.append([Mean_Y,Std_Y])
    with open('./NN_Log/Log_%i.txt' % (Gen),'a') as log_file:
        log_file.write('Data successfully normalized\n')
        log_file.write('X norm factor: %f\n' % X_Max)
        log_file.write('Y norm factors MEAN:')
        for item in Y_norms:
            log_file.write(' %f' % item[0])
        log_file.write('\nY norm factors STD:')
        for item in Y_norms:
            log_file.write(' %f' % item[1])
        log_file.write('\n')

    #generate Model
    Models = []
    if args.reload:
        for i in range(States_to_use):
            Models.append(reload_model(i,Gen))
    else:
        if args.state == 0: #run all states
            for i in range(States_to_use):
                Models.append(Neural_Network(i))
        else: #run only a specific state
            Model = Neural_Network(args.state-1)
            save_model(Model,args.state-1,Gen)
            sys.exit() #This Method requires a restart with reload!

    #Save Model
    if not args.no_save:
        for i in range(States_to_use):
            save_model(Models[i],i,Gen)

    if args.evaluate:
        if not os.path.isdir('Evaluate'):
            os.mkdir('Evaluate')
        os.chdir('./Evaluate')
        #Evaluate Model
        print('Evaluating model')
        for m, model in enumerate(Models):
            Y = States_Y[m]
            Mean_Y = Y_norms[m][0]
            Std_Y = Y_norms[m][1]
            Y = Y * Std_Y + Mean_Y
            Y_Test = States_Y_Test[m]
            Y_Test = Y_Test * Std_Y + Mean_Y
            Y_predicted = model.predict(X)
            Y_predicted = Y_predicted.reshape(Y_predicted.shape[0]).reshape(Y_predicted.shape[0])
            Y_predicted = Y_predicted * Std_Y + Mean_Y
            C_Train = numpy.corrcoef(Y,Y_predicted)[0,1]
            Y_test_predicted = model.predict(X_Test)
            Y_test_predicted = Y_test_predicted.reshape(Y_test_predicted.shape[0]).reshape(Y_test_predicted.shape[0])
            Y_test_predicted = Y_test_predicted * Std_Y + Mean_Y
            C_Test = numpy.corrcoef(Y_Test,Y_test_predicted)[0,1]
            col = ['#f49f1c','#3686bf','#2936cc','green']
            if args.single:
                #Normalize the REU to protein size
                Y /= Total_States
                Y_predicted /= Total_States
                Y_Test /= Total_States
                Y_test_predicted /= Total_States
            plt.plot(Y,Y_predicted,'.',label='Training',color=col[0],markersize=2)
            plt.plot(Y_Test,Y_test_predicted,'.',label='Testing',color=col[1],markersize=2)
            plt.plot(Y,Y,'-',color='grey',label='_nolegend_',markersize=1)
            plt.plot(Y_Test,Y_Test,'-',color='grey',label='_nolegend_',markersize=1)
            plt.xlabel('True Score [REU]')
            plt.ylabel('Predicted Score [REU]')
            plt.title('Rosetta:MSF$_{NN}$',y=1.03)
            lgnd = plt.legend()
            for i in range(2):
                lgnd.legendHandles[i]._legmarker.set_markersize(12)
            lgnd.get_frame().set_linewidth(0.0)
            plt.savefig('Performance_State_%i.png' % (m+1),dpi=1800)
            plt.close()
            print('The correlation coefficients are: %.3f for training and %.3f for testing' % (C_Train,C_Test))
            Errors_Y = abs(Y - Y_predicted)
            Avg_Error_Y = numpy.mean(Errors_Y)
            print('Average error training: %f' % Avg_Error_Y)
            Errors_Y_Test = abs(Y_Test - Y_test_predicted)
            Avg_Error_Test_Y = numpy.mean(Errors_Y_Test)
            print('Average error testing: %f' % Avg_Error_Test_Y)
        os.chdir('..')

    #Generating new Sequences
    AA_pool = 'RHKDECGPSTNQAVILMFYW'
    print('Generating Score_Dic')
    Score_Dic = Get_Scores()
    Values = Score_Dic.keys()
    Values.sort() # small to large
    Seeds = []
    for i in range(10):
        Seeds.append(Score_Dic[Values[i]])
    Used_Seq = []
    with open('./NN_Log/Log_%i.txt' % (Gen),'a') as log_file:
        for S, Seed in enumerate(Seeds):
            log_file.write('Seed_%d: %s\n' % ((S+1), Seed))
        log_file.write('New_Population: %d\n' % args.pop)
    if not args.evaluate_only: #Predict new Sequences
        Operators = args.operators
        # If len == 1, always use this operator
        if len(Operators) == 1:
            Operators *= (len(Models)-1)
        Used_Seq = Score_Dic.values()
        To_write = []
        # New sequences
        All_Top_Sequences = []
        All_Top_Scores = []
        for i in range(args.random_seq_split): # 20* 1E5 reduce extensive mem usage
            print('Running generation round %d' % (i+1))
            Sequences = random_seq(Seeds,AA_pool)
            Encoded_Seq = []
            for Sequence in Sequences:
                Tempsequence = encode(Encoding_Dic,Sequence)
                Encoded_Seq.append(Tempsequence)
            X = pandas.DataFrame(Encoded_Seq)
            X = X.apply(lambda a: a / float(X_Max))
            for m, model in enumerate(Models):
                Y_predicted = model.predict(X)
                Y_predicted = Y_predicted.reshape(Y_predicted.shape[0]).reshape(Y_predicted.shape[0])
                Mean_Y = Y_norms[m][0]
                Std_Y = Y_norms[m][1]
                Y_predicted = Y_predicted * Std_Y + Mean_Y
                if m == 0:
                    Y_predicted_total = Y_predicted
                else:
                    if Operators[m-1] == '+':
                        Y_predicted_total = Y_predicted_total + Y_predicted
                    elif Operators[m-1] == '-':
                        Y_predicted_total = Y_predicted_total - Y_predicted
                    else:
                        sys.exit('UNKNOWN OPERATOR')
            sorter = numpy.argsort(Y_predicted_total) #no reverse since original scores small to large
            j = -1
            New_Seq_Counter = 0
            while New_Seq_Counter <= args.pop:
                j += 1
                if j > len(sorter):
                    with open('./NN_Log/Log_%i.txt' % (Gen),'a') as log_file:
                        log_file.write('Out of sequences at mem save\n')
                    break
                if Sequences[sorter[j]] in Used_Seq:
                    continue
                All_Top_Sequences.append(Sequences[sorter[j]])
                All_Top_Scores.append(Y_predicted_total[sorter[j]])
                New_Seq_Counter += 1
            Sequences = []

        Sequences = All_Top_Sequences #write back mem fix
        Y_predicted_total = All_Top_Scores #write back mem fix
        sorter = numpy.argsort(All_Top_Scores) #write back mem fix

        counter = 0
        for idx in sorter:
            if Sequences[idx] not in Used_Seq:
                counter += 1
                To_write.append([counter,Sequences[idx], Y_predicted_total[idx]])
                Used_Seq.append(Sequences[idx])
            if len(To_write) == pop_size:
                break
        else:
            with open('./NN_Log/Log_%i.txt' % (Gen),'a') as log_file:
                log_file.write('Out of sequences at function\n')

        with open('./NN_Log/Log_%i.txt' % (Gen),'a') as log_file:
            print('writing sequences')
            log_file.write('CSV marker line\n')
            log_file.write('generation %d\n' % (Gen +1))
            for item in To_write:
                log_file.write('%d %s %f\n' % (item[0],item[1],item[2]))

if __name__ == '__main__':
    main()
