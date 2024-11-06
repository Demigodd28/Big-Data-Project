import pandas as pd

# def get_keyword(quest):
#     table = {
#         "地點時間":["本部", "分部", "地點", "地方", "時段", "時間", "什麼時候"], 
#         "授課教師":["名字", "什麼教授", "什麼老師", "叫什麼", "是誰"],
#         "課程學分數量":["學分"],
#         "課程名稱":["什麼課"],
#         "限修條件":["限修條件", "擋修"],
#         "限修人數":["限修人數"],
#     }
#     temp = [[], []]
#     for index, (row, items) in enumerate(table.items()):
#         for item in items:
#             if item in quest:
#                 temp[0].append(index)
#                 temp[1].append(row)
#     if temp == [[], []]:
#         temp = [[0], [None]]
#     return temp # temp = [[2], [sss]]    

def filtered_question(index, arg):## index:問什麼 arg:關鍵字
    if index == 0:
        return 'No data'
    
    df = pd.read_csv("modified_ee.csv")
    if index == 1:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['地點時間'][6:]##place
    if index == 2:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['地點時間'][0:5]##time
    if index == 3:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['授課教師']
    if index == 4:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['課程學分數量']
    if index == 5:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['限修條件']
    if index == 6:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['全英語']
    if index == 7:
        row = df[df['中文課程名稱'] == arg]
        if not row.empty:
            return row.iloc[0]['備註']
    if index == 8:
        row = df[df['授課教師'] == arg]
        if not row.empty:
            return row['授課教師'].unique()
    if index == 9:
        row = df[df['地點時間'][0:5] == arg]
        if not row.empty:
            return row['中文課程時間'].unique()
    if index == 10:
        row = df[df['課程學分數量'] == arg]
        if not row.empty:
            return row['中文課程時間'].unique()

if __name__ == '__main__':
    print(filtered_question(2, "數位系統"))