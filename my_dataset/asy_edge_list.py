import numpy as np
import pandas as pd

from tqdm import tqdm

tripleDDI = pd.read_csv('./DrugSMILES/ourdrug-interaction-#drug.csv', sep='\t', header=None)
tripleDDI_numpy = tripleDDI.to_numpy()


# relationID = pd.read_csv('./DrugSMILES/ourRelationType.csv', sep='\t', header=None)
# relationID_numpy = relationID.to_numpy()

relationID = pd.read_csv('./DrugSMILES/my_relation_real.csv', sep='\t', header=None)
relationID_numpy = relationID.to_numpy()


# drugID = pd.read_csv('./DrugSMILES/ourDrugSMILES.csv', sep='\t', header=None)
# drugID_numpy = drugID.to_numpy()

drugID = pd.read_csv('./DrugSMILES/my_drug_real.csv', sep='\t', header=None)
drugID_numpy = drugID.to_numpy().flatten()


# f_edge_list = open('./DrugSMILES/my_edge_list.csv','w',encoding='utf-8')
#
# f_edge_type = open('./DrugSMILES/my_edge_type.csv','w',encoding='utf-8')

for i in tqdm(range(len(tripleDDI_numpy))):


    idDrug1 = None
    idDrug2 = None
    tripleDDI_i = tripleDDI_numpy[i]


    for dID in range(len(drugID_numpy)):
        if tripleDDI_i[0] == drugID_numpy[dID]:
            idDrug1 = dID
        if tripleDDI_i[1] == drugID_numpy[dID]:
            idDrug2 = dID

        if idDrug1 == 507 and idDrug2 == 1080:
            print(tripleDDI_numpy[i])

    if idDrug1 != None and idDrug2 != None:
        edge_index_lines = '\t'.join([str(idDrug1), str(idDrug2)])
        # f_edge_list.writelines(edge_index_lines + '\n')

        for rID in range(len(relationID_numpy)):
            if tripleDDI_i[2] in relationID_numpy[rID]:
                idRela = rID

                # f_edge_type.writelines(str(idRela) + '\n')
                break







# f_edge_list.close()
# f_edge_type.close()