from rdkit.Chem import AllChem
from rdkit import Chem


m1 = Chem.MolFromSmiles('CCCC(N)=O')

m1 = Chem.MolFromSmiles('[Li]Cl')
fp1 = AllChem.GetMorganFingerprint(m1,2,useCounts=True)
# <rdkit.DataStructs.cDataStructs.UIntSparseIntVect object at 0x...>
# 得到一个需要转换的 UIntSparseIntVect（整数向量）
fp2= AllChem.GetMorganFingerprintAsBitVect(m1,2,nBits=1024)
#<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x...>
# 得到一个需要转换的ExplicitBitVect（1024位向量）
fp3= AllChem.GetMorganFingerprintAsBitVect(m1,2)
#<rdkit.DataStructs.cDataStructs.ExplicitBitVect object at 0x...>
# 得到一个需要转换的ExplicitBitVect（2048位向量）
fp2.ToBitString() #00000000，将fp2转换为位串
fp3.ToBitString()  #00000000，将fp3转换为位串

print(fp2.ToBitString())

print(len(fp2.ToBitString()))

print('fp3.ToBitString()_length:{}'.format(len(fp3.ToBitString())))
