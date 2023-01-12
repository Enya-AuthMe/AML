import pandas as pd
from tqdm import tqdm

A_path = './訓練資料集_first/public_train_x_custinfo_full_hashed.csv'
B_path = './訓練資料集_first/public_train_x_ccba_full_hashed.csv'
A = pd.read_csv(A_path)
B = pd.read_csv(B_path)

obj_name = 'cust_id'


# # 檢測 custinfo 對應其他表格是否互相個體相同
# cnt = 0
# ID = []
# for i, id in tqdm(enumerate(B[obj_name])):
#     if A[A[obj_name] == id].index.values.size != 0:
#         cnt += 1
#     else:
#         ID.append(id)


# # 檢測重複
obj = A[obj_name]
obj = obj.append(B[obj_name], ignore_index=True)
obj_bool = obj.duplicated(keep=False)

num = 0
true_num = 0
for i, bool in tqdm(enumerate(obj_bool)):
    if bool == False:
        num += 1
    elif bool == True:
        true_num += 1

print('False  ', num)
print('True   ', true_num)
print('total=A+B  ', i+1, len(A), len(B))

breakpoint()
