import pandas as pd

count = 0
empty = []

for x in range(7128):
    empty.append([x,"-",0,0,0,0])

def rmduplicated(memlist):
    for i, row in enumerate(memlist):
        if memlist[i-1][0] == row[0]:
            avg = (memlist[i-2][2] + memlist[i-3][2] + memlist[i-4][2] + memlist[i-5][2] + memlist[i-6][2])/5

            if abs(memlist[i-1][2] - avg) < abs(row[2] - avg):
                memlist.remove(row) #i행 삭제
            else:
                memlist.remove(memlist[i-1]) #i-1행 삭제


def frame(df, memlist):
    for row in memlist:
        df.iloc[row[0],2] = row[2]
        df.iloc[row[0],3] = row[3]
        df.iloc[row[0],4] = row[4]
        df.iloc[row[0],5] = row[5]


# 멤버별 초기값 0, "-", 911, 178, 107, 192 sana기준 - 1080p
def initial(df):
    df.iloc[0,2] = 1821
    df.iloc[0,3] = 358
    df.iloc[0,4] = 218
    df.iloc[0,5] = 381


# 초기 frame linear 증가
def linear_frame(df):
    linear_x = 0
    linear_y = 0
    linear_w = 0
    linear_h = 0
    end = 0
    
    for idx, row in df.iterrows():
        if idx == 0:
            pass
        
        if row["x"] != 0:
            linear_x = (row["x"] - df.iloc[0,2])/idx
            linear_y = (row["y"] - df.iloc[0,3])/idx
            linear_w = (row["w"] - df.iloc[0,4])/idx
            linear_h = (row["h"] - df.iloc[0,5])/idx
            end = idx
                
    for idx, row in df.iterrows():
        if idx == end:
            break
            
        if row["x"] == 0:
            df.iloc[idx,2] = round(df.iloc[0,2] + linear_x*idx)
            df.iloc[idx,3] = round(df.iloc[0,3] + linear_y*idx)
            df.iloc[idx,4] = round(df.iloc[0,4] + linear_w*idx)
            df.iloc[idx,5] = round(df.iloc[0,5] + linear_h*idx)


# 튀는값 제거 
def moving_avg_frame(df): 
    sumlist_x = []
    sumlist_y = []
    sumlist_w = []
    sumlist_h = []
    for i in range(6):
        sumlist_x.append(df.iloc[i,2])
        sumlist_y.append(df.iloc[i,3])
        sumlist_w.append(df.iloc[i,4])
        sumlist_h.append(df.iloc[i,5])

    for idx, row in df.iterrows():
        if idx < 6: # 7번째 frame부터
            pass
        
        tmp_x = sum(sumlist_x)/len(sumlist_x) # 평균값
        tmp_y = sum(sumlist_y)/len(sumlist_y)
        tmp_w = sum(sumlist_w)/len(sumlist_w)
        tmp_h = sum(sumlist_h)/len(sumlist_h)
        
        if abs(df.iloc[idx,2] - tmp_x) >= 100:
            df.iloc[idx,2] = tmp_x
            df.iloc[idx,3] = tmp_y
            df.iloc[idx,4] = tmp_w
            df.iloc[idx,5] = tmp_h

            sumlist_x.remove(sumlist_x[0])
            sumlist_y.remove(sumlist_y[0])
            sumlist_w.remove(sumlist_w[0])
            sumlist_h.remove(sumlist_h[0])
            sumlist_x.append(tmp_x)
            sumlist_y.append(tmp_y)
            sumlist_w.append(tmp_w)
            sumlist_h.append(tmp_h)
            
        else:
            sumlist_x.remove(sumlist_x[0])
            sumlist_y.remove(sumlist_y[0])
            sumlist_w.remove(sumlist_w[0])
            sumlist_h.remove(sumlist_h[0])
            sumlist_x.append(row["x"])
            sumlist_y.append(row["y"])
            sumlist_w.append(row["w"])
            sumlist_h.append(row["h"])         


def to_int(df):
    df[["x","y","w","h"]] = df[["x","y","w","h"]].astype(int)


# h 1920 x w 1080
# x + w/2 - 540, y + h/2 - 9

# 2085+170 -540 , 305+205 - 250
# x, y, w, h, n1 - 0, n2 - 0, n3 - 395, n4 - 633
def reshape_frame(df):    
    for idx, row in df.iterrows():
        if (int(row["x"] + row["w"]/2 - 540)+1080) > 3840:
            df.iloc[idx,2] = 2760
        else:
            if int(row["x"] + row["w"]/2 - 540) < 0:
                df.iloc[idx,2] = 0
            else:
                df.iloc[idx,2] = int(row["x"] + row["w"]/2 - 540) # 3840
        
        if (int(row["y"] - row["h"]*0.1)+1920) > 2160:
            df.iloc[idx,3] = 240
        else:
            df.iloc[idx,3] = int(row["y"] - row["h"]*0.1) # 2160
        
        df.iloc[idx,4] = 1080
        df.iloc[idx,5] = 1920


def add_frame(df):
    del df["frame"]
    del df["name"]
    df["n1"] = 0
    df["n2"] = 0
    df["n3"] = 395
    df["n4"] = 633


def write_csv(df, name):
    df.to_csv(name, index=False, header=False)


def main():
    data = pd.read_csv('loc_1080_0211.csv',header=None)
    data.columns=['frame','name','x','y','w','h']

    member = ["tzuyu","nayeon","jeongyeon","dahyun","sana","jihyo","momo","mina","chaeyoung"]
    person = ["person","person","person","person","person","person","person","person","person"]

    temp = []
    ptemp = []
    tzuyu,nayeon,jeongyeon,dahyun,sana,jihyo,momo,mina,chaeyoung = [], [], [], [], [], [], [], [], []

    for idx, rw in data.iterrows():
        if rw["name"] != "person":
            temp.append([rw["frame"], rw["name"], rw["x"], rw["y"], rw["w"], rw["h"]])
        else:
            ptemp.append([rw["frame"], rw["name"], rw["x"], rw["y"], rw["w"], rw["h"]])

    for i, x in enumerate(temp):
        if x[1] == "tzuyu":
            tzuyu.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "nayeon":
            nayeon.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "jeongyeon":
            jeongyeon.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "dahyun":
            dahyun.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "sana":
            sana.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "jihyo":
            jihyo.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "momo":
            momo.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "mina":
            mina.append([x[0], x[1], x[2], x[3], x[4], x[5]])
        elif x[1] == "chaeyoung":
            chaeyoung.append([x[0], x[1], x[2], x[3], x[4], x[5]])

    rmduplicated(tzuyu)
    rmduplicated(nayeon)
    rmduplicated(jeongyeon)
    rmduplicated(dahyun)
    rmduplicated(sana)
    rmduplicated(jihyo)
    rmduplicated(momo)
    rmduplicated(mina)
    rmduplicated(chaeyoung)

    cf1 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf2 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf3 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf4 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf5 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf6 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf7 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf8 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])
    cf9 = pd.DataFrame(empty, columns=["frame","name","x","y","w","h"])

    cf1["name"] = "tzuyu"
    cf2["name"] = "nayeon"
    cf3["name"] = "jeongyeon"
    cf4["name"] = "dahyun"
    cf5["name"] = "sana"
    cf6["name"] = "jihyo"
    cf7["name"] = "momo"
    cf8["name"] = "mina"
    cf9["name"] = "chaeyoung"
            
    frame(cf1, tzuyu)
    frame(cf2, nayeon)
    frame(cf3, jeongyeon)
    frame(cf4, dahyun)
    frame(cf5, sana)
    frame(cf6, jihyo)
    frame(cf7, momo)
    frame(cf8, mina)
    frame(cf9, chaeyoung)

    initial(cf1)
    initial(cf2)
    initial(cf3)
    initial(cf4)
    initial(cf5)
    initial(cf6)
    initial(cf7)
    initial(cf8)
    initial(cf9)
            
    linear_frame(cf1)
    linear_frame(cf2)
    linear_frame(cf3)
    linear_frame(cf4)
    linear_frame(cf5)
    linear_frame(cf6)
    linear_frame(cf7)
    linear_frame(cf8)
    linear_frame(cf9)

    moving_avg_frame(cf1)
    moving_avg_frame(cf2)
    moving_avg_frame(cf3)
    moving_avg_frame(cf4)
    moving_avg_frame(cf5)
    moving_avg_frame(cf6)
    moving_avg_frame(cf7)
    moving_avg_frame(cf8)
    moving_avg_frame(cf9)

    to_int(cf1)
    to_int(cf2)
    to_int(cf3)
    to_int(cf4)
    to_int(cf5)
    to_int(cf6)
    to_int(cf7)
    to_int(cf8)
    to_int(cf9)

    reshape_frame(cf1)
    reshape_frame(cf2)
    reshape_frame(cf3)
    reshape_frame(cf4)
    reshape_frame(cf5)
    reshape_frame(cf6)
    reshape_frame(cf7)
    reshape_frame(cf8)
    reshape_frame(cf9)

    add_frame(cf1)
    add_frame(cf2)
    add_frame(cf3)
    add_frame(cf4)
    add_frame(cf5)
    add_frame(cf6)
    add_frame(cf7)
    add_frame(cf8)
    add_frame(cf9)

    write_csv(cf1,'tzuyu.csv')
    write_csv(cf2,'nayeon.csv')
    write_csv(cf3,'jeongyeon.csv')
    write_csv(cf4,'dahyun.csv')
    write_csv(cf5,'sana.csv')
    write_csv(cf6,'jihyo.csv')
    write_csv(cf7,'momo.csv')
    write_csv(cf8,'mina.csv')
    write_csv(cf9,'chaeyoung.csv')

if __name__ == "__main__":
    main()