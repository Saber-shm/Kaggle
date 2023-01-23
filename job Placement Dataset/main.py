import tkinter as tk
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
root = tk.Tk()
lb_ssc_board = LabelEncoder()
lb_hsc_board  = LabelEncoder()
lb_hsc_suject = LabelEncoder()
lb_undergrad_degree = LabelEncoder()
lb_work_experience = LabelEncoder()
lb_specialisation = LabelEncoder()
lb_status = LabelEncoder()
df = pd.read_csv('Job_Placement_Data.csv')
def preprocessing(df,lb_ssc_board,lb_hsc_board,lb_hsc_suject,lb_undergrad_degree,lb_work_experience,lb_specialisation,lb_status):
    df.drop("gender",axis = 1,inplace = True)
    df["ssc_board"] = lb_ssc_board.fit_transform(df['ssc_board'])
    df["hsc_board"] = lb_hsc_board.fit_transform(df["hsc_board"])
    df['hsc_subject'] = lb_hsc_suject.fit_transform(df["hsc_subject"])
    df["undergrad_degree"] = lb_undergrad_degree.fit_transform(df["undergrad_degree"])
    df["work_experience"] =  lb_work_experience.fit_transform(df["work_experience"])
    df["specialisation"] = lb_specialisation.fit_transform(df['specialisation'])
    df["status"] =lb_status.fit_transform(df["status"])
    return df

def spliting(df):
    X = df.drop("status",axis = 1)
    y = df["status"]
    xtrain,xtest,ytrain,ytest = train_test_split(X,y, test_size= 0.3, random_state= 4)
    return xtrain,xtest,ytrain,ytest
def training_model(xtrain,ytrain,xtest,ytest):
    model = RandomForestClassifier()
    model.fit(xtrain,ytrain,)
    ypred = model.predict(xtest)
    accuracy = accuracy_score(ytest,ypred)
    return model,accuracy
def preprocessing_predicting_data(gender_entry,ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry):
    r = [ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry]
    dic = {
        "ssc_percentage":[],
        "ssc_board":[],
        "hsc_percentage":[],
        "hsc_board":[],
        "hsc_subject":[],
        "degree_percentage":[],
        "undergrad_degree":[],
        "work_experience":[],
        "emp_test_percentage":[],
        "specialisation":[],
        'mba_percent':[]
    }
    s = list(dic.keys())
    for i in range(len(list(s))):
        dic[s[i]].append(r[i].get())
    df = pd.DataFrame(dic)

    df["ssc_board"] = lb_ssc_board.fit_transform(df['ssc_board'])
    df["hsc_board"] = lb_hsc_board.fit_transform(df["hsc_board"])
    df['hsc_subject'] = lb_hsc_suject.fit_transform(df["hsc_subject"])
    df["undergrad_degree"] = lb_undergrad_degree.fit_transform(df["undergrad_degree"])
    df["work_experience"] =  lb_work_experience.fit_transform(df["work_experience"])
    df["specialisation"] = lb_specialisation.fit_transform(df['specialisation'])

    return df



def submit(gender_entry,ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry,preproced_df,xtrain,xtest,ytrain,ytest,model):

    preproced_predicting_data = preprocessing_predicting_data(gender_entry,ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry)
    result = model.predict(preproced_predicting_data)
    result = lb_status.inverse_transform(result)
    rr = tk.Label(root,text = str(result))
    rr.pack()

preproced_df = preprocessing(df,lb_ssc_board,lb_hsc_board,lb_hsc_suject,lb_undergrad_degree,lb_work_experience,lb_specialisation,lb_status)
xtrain,xtest,ytrain,ytest = spliting(preproced_df)
model,accuracy = training_model(xtrain,ytrain,xtest,ytest)



"""
--  ------               --------------  -----  
 0   gender               215 non-null    object 
 1   ssc_percentage       215 non-null    float64
 2   ssc_board            215 non-null    object 
 3   hsc_percentage       215 non-null    float64
 4   hsc_board            215 non-null    object 
 5   hsc_subject          215 non-null    object 
 6   degree_percentage    215 non-null    float64
 7   undergrad_degree     215 non-null    object 
 8   work_experience      215 non-null    object 
 9   emp_test_percentage  215 non-null    float64
 10  specialisation       215 non-null    object 
 11  mba_percent          215 non-null    float64
 12  status               215 non-null    object 

"""
root.geometry("700x600")

gender_label = tk.Label(root,text = 'Gender (M,F):')
gender_entry = tk.Entry(root)

ssc_percentage_label = tk.Label(root,text = "SSC percentage: " )
ssc_percentage_entry = tk.Entry(root)

ssc_board_label = tk.Label(root,text = "SSC board: " )
ssc_board_entry = tk.Entry(root)

hsc_percentage_label = tk.Label(root,text = "HSC percentage: " )
hsc_percentage_entry = tk.Entry(root)

hsc_board_abel = tk.Label(root,text = "HSC board : " )
hsc_board_entry = tk.Entry(root)



hsc_subject_label = tk.Label(root,text = "HSC subject: " )
hsc_subject_entry = tk.Entry(root)



degree_percentage_label = tk.Label(root,text = "Degree percentage: " )
degree_percentage_entry = tk.Entry(root)




undergrad_degree_label = tk.Label(root,text = "Undergrad degree: " )
undergrad_degree_entry = tk.Entry(root)

work_experience_label = tk.Label(root,text = "Work experience: " )
work_experience_entry = tk.Entry(root)

emp_test_percentage_label = tk.Label(root,text = "EMP test percentage: " )
emp_test_percentage_entry = tk.Entry(root)

specialisation_label = tk.Label(root,text = "Specialisation : " )
specialisation_entry = tk.Entry(root)

mba_percent_label = tk.Label(root,text = "MBA percent: " )
mba_percent_entry = tk.Entry(root)

submit_button = tk.Button(root,text = "Submit",command = lambda: submit(gender_entry,ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry,preproced_df,xtrain,xtest,ytrain,ytest,model))
to_pack = [gender_label,gender_entry,ssc_percentage_label,ssc_percentage_entry,ssc_board_label,ssc_board_entry,hsc_percentage_label,hsc_percentage_entry,hsc_board_abel,hsc_board_entry,
hsc_subject_label,hsc_subject_entry,degree_percentage_label,degree_percentage_entry,undergrad_degree_label,undergrad_degree_entry,
work_experience_label,work_experience_entry,emp_test_percentage_label,emp_test_percentage_entry,specialisation_label,specialisation_label,specialisation_entry,
mba_percent_label,mba_percent_entry,submit_button
]
x = preprocessing_predicting_data(gender_entry,ssc_percentage_entry,ssc_board_entry,hsc_percentage_entry,hsc_board_entry,hsc_subject_entry,degree_percentage_entry,undergrad_degree_entry,work_experience_entry,emp_test_percentage_entry,specialisation_entry,mba_percent_entry)
print(x)
for i in to_pack:
    i.pack()
root.mainloop()
