import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import messagebox
import time
import tkinter.ttk as ttk
import warnings
import platform
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD
import random

warnings.filterwarnings(action='ignore')

# 한글폰트 적용
if platform.system() == 'Darwin':  # 맥
    plt.rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False  # 한글 폰트 사용시 마이너스 폰트 깨짐 해결

global window
window = Tk()
window.title("바른 우리말 학습")  # 화면의 제목 지정
window.geometry("360x540+500+60")  # 화면 크기 및 나타나는 위치 고정
window.resizable(False, False)  # 화면 크기 변경 못하게 설정

problem_df = pd.read_excel('일일학습_문제정리_엑셀.xlsx')
problem_pd = pd.read_excel('수준테스트문제.xlsx')

# User_info = pd.read_excel('사용자 정보2.xlsx')

global cnt_O
cnt_O = 0


# 공통 기능 부모 클래스
class FUNC:
    """Super Class"""
    global cnt_O
    cnt_O = 0

    global answer, result, solving
    answer = []
    result = []
    solving = []

    global SV
    SV = ""

    global i
    i = -1
    global next
    next = True

    n = 20

    def __init__(self, n):
        self.n = n

    label_frame = Frame(window, bg='blue')
    label_frame.pack(fill='both', expand=1)
    problem_frame = Frame(window, bg='yellow')
    problem_frame.pack(side=TOP, fill='both', expand=1)
    etc_frame = Frame(window, bg='green')
    etc_frame.pack(side=BOTTOM, fill='both', expand=1)

    def marking(self, problem_type):
        global answer, result, cnt_O
        self.problem_frame.destroy()
        self.problem_frame = Frame(window, bg='yellow')
        self.problem_frame.pack(side=TOP, fill='both', expand=1)

        for a in range(len(answer)):
            if answer[a] == problem_type['정답'][a]:
                # result.append('O')
                text = str(a + 1) + '.  0'
                cnt_O += 1
                msg = Message(self.problem_frame, text=text, width=320)
                msg.pack()
            else:
                # result.append('X')
                text = str(a + 1) + '.  X'
                msg = Message(self.problem_frame, text=text, width=320)
                msg.pack()

    def progress_bar(self, problem_type):
        self.etc_frame.destroy()
        self.problem_frame.destroy()
        self.problem_frame = Frame(window, bg='yellow')
        self.problem_frame.pack(side=TOP, fill='both', expand=1)

        p_var2 = DoubleVar()
        progressbar = ttk.Progressbar(self.problem_frame, maximum=100, length=150, variable=p_var2)
        progressbar.pack(pady=20, anchor=CENTER, ipadx=100)
        msg = Message(self.problem_frame, text='사용자의 답안을 채점 중입니다.', font=('Helvetica', 15), width=400)
        msg.pack()

        for i in range(1, 101):
            time.sleep(0.01)

            p_var2.set(i)
            progressbar.update()

        OK_button = Button(self.problem_frame, text='확인', command=lambda: self.marking(problem_type))
        OK_button.pack(padx=115, pady=150)

    def next_Select(self):
        self.problem_frame.destroy()
        #####
        self.problem_frame = Frame(window, bg='yellow')
        self.problem_frame.pack(side=TOP, fill='both', expand=1)

    def next_Q(self):
        global SV, answer
        if SV.get() == "":
            global next
            next = False
            messagebox.showinfo("Error.", "정답을 입력해주세요")
        else:
            next = True
            answer.append(SV.get())
        self.problem()

    def problem(self, n, problem_type):
        global SV
        SV = StringVar()
        global next
        global i

        if next is True:
            i += 1

        if i == n:
            self.progress_bar()
            return

        self.problem_frame.destroy()
        self.problem_frame = Frame(window, bg='yellow')
        self.problem_frame.pack(side=TOP, fill='both', expand=1)

        self.etc_frame.destroy()
        self.etc_frame = Frame(window, bg='green')
        self.etc_frame.pack(side=BOTTOM, fill='both', expand=1)

        msg_q = Message(self.problem_frame, text=str(i+1) + '. ' + problem_type['문제'][i], font=('bold', 15), width=320)
        msg_q.pack(pady=80)
        textbox = Entry(self.problem_frame, textvariable=SV)
        textbox.pack()

        next_button = Button(self.etc_frame, text='다음', command=self.next_Q)
        if i == (n - 1):
            next_button = Button(self.etc_frame, text='답안 제출', command=self.next_Q)
        next_button.pack(padx=115, pady=100)


global LEVEL


# 수준 테스트
class Level_Test(FUNC):

    def next_Select(self):
        super().next_Select()
        instance = LearningManager()
        instance.selectMode()

    def marking(self, problem_type):
        global cnt_O, LEVEL
        super().marking(problem_pd)
        print(cnt_O)
        if cnt_O >= 15:
            LEVEL = '상'
        elif cnt_O >= 10:
            LEVEL = '중'
        else:
            LEVEL = '하'

        text = '당신의 수준은 *' + LEVEL + '* 입니다.'
        msg = Message(self.problem_frame, text=text, font=('Helvetica', 15), width=320)
        msg.pack(pady=10)

        # 유저 정보 등록

        OK_button = Button(self.problem_frame, text='확인', command=self.next_Select)
        OK_button.pack(side=RIGHT)

    def progress_bar(self):
        super().progress_bar(problem_pd)

    def next_Q(self):
        super().next_Q()

    def problem(self):
        super().problem(20, problem_pd)

    def start_Click(self):
        self.label_frame.destroy()
        global cnt_O
        cnt_O = 0
        msg = Message(self.problem_frame, text='진짜 시작하겠습니다.', font=('Helvetica', 15), width=400)
        msg.pack(padx=50, pady=80)
        next_button = Button(self.etc_frame, text='다음', command=self.problem)
        next_button.pack(padx=115, pady=100)

    def level_test(self):
        start_label = Label(self.label_frame, text='수준 테스트를 시작합니다.', font=('Helvetica'))
        start_label.pack(padx=50, pady=75)

        start_button = Button(self.label_frame, text='시작', command=self.start_Click)
        start_button.pack(padx=115, pady=150)


# 학습 모드 선택
class LearningManager:
    mode_frame = Frame(window)  # 학습 모드 선택 프레임
    daily_frame = Frame(window)  # 일일학습 선택 시 프레임

    def next_First(self, day):
        self.daily_frame.destroy()
        # self.daily_frame.pack_forget()
        instance = RecommendProblem(15, day)
        instance.firstDay(0)

    def next_Other(self, day):
        self.daily_frame.destroy()
        # self.daily_frame.pack_forget()
        instance = RecommendProblem(7, day)
        instance.analyzeTaste(0)

    def runDailylearning(self):
        # self.mode_frame.pack_forget()
        self.mode_frame.destroy()

        self.daily_frame = Frame(window)
        self.daily_frame.pack()
        Day1_btn = Button(self.daily_frame, text="1 일차", width=10, command=lambda: self.next_First(1))
        Day1_btn.pack(padx=125, pady=70)
        Day2_btn = Button(self.daily_frame, text="2 일차", width=10, command=lambda: self.next_Other(2))
        Day2_btn.pack(padx=125, pady=60)
        Day3_btn = Button(self.daily_frame, text="3 일차", width=10, command=lambda: self.next_Other(3))
        Day3_btn.pack(padx=125, pady=70)
        print('일일 학습 실행')

    def runOther(self):
        self.mode_frame.destroy()
        self.mode_frame = Frame(window)
        self.daily_frame = Frame(window)
        self.daily_frame.pack()
        prev_btn = Button(self.daily_frame, text="이전", command=self.selectMode)
        prev_btn.pack(padx=115, pady=100)
        print('일일 학습 제외한 학습 실행')

    def End(self):
        exit(0)

    def selectMode(self):
        global i, answer, cnt_O
        i = -1
        answer = []
        cnt_O = 0

        self.daily_frame.destroy()
        ###
        self.mode_frame = Frame(window)
        ##
        self.mode_frame.pack()
        select_txt = Label(self.mode_frame, text="학습 모드 선택", font=('Helvetica', 20, 'bold'))
        select_txt.pack(padx=85, pady=50)

        btn_Daily = Button(self.mode_frame, text="일일 학습", bg='yellow', width=15, command=self.runDailylearning)
        btn_Daily.pack(padx=125, pady=30)
        btn_Sapre = Button(self.mode_frame, text="자투리 학습", bg='orange', width=15, command=self.runOther)
        btn_Sapre.pack(padx=125, pady=30)
        btn_Review = Button(self.mode_frame, text="복습 하기", bg='sky blue', width=15, command=self.runOther)
        btn_Review.pack(padx=125, pady=30)
        btn_Vocabulary = Button(self.mode_frame, text="단어장", bg='pink', width=15, command=self.runOther)
        btn_Vocabulary.pack(padx=125, pady=30)
        btn_Terminal = Button(self.mode_frame, text="종료", bg='red', width=15, command=self.End)
        btn_Terminal.pack(padx=160, pady=20)

class RecommendProblem(FUNC):
    n = 7
    day = 2
    def __init__(self, n, day):
        self.n = n
        self.day = day

    def makeMatrix(self):
        User_info = pd.read_excel('사용자 정보2.xlsx')
        df_Usermatrix = User_info[['유저 아이디', '문제 번호', '수준', '맞춤법 선호도']]
        for i in range(len(df_Usermatrix) - 1):
            if problem_df['유형'][df_Usermatrix['문제 번호'][i]] == '띄어쓰기':
                df_Usermatrix['맞춤법 선호도'][i] = User_info['띄어쓰기 선호도'][i]

        # df_pivot_Usermatrix = df_Usermatrix.pivot_table('맞춤법 선호도', index='유저 아이디', columns='문제 번호').fillna(0)
        return df_Usermatrix

    learning_frame = Frame(window, bg='pink')  # 학습 프레임
    day_label = Label(learning_frame, text=day, font=('Helvetica', 18, "bold"))
    differ_button = Button(learning_frame, text='다른 유형')

    global LEVEL

    def firstDay(self, day):
        global LEVEL
        # LEVEL = '하'  ###

        self.learning_frame = Frame(window, bg='pink')
        self.learning_frame.pack(fill='x')
        day = str(self.day) + '일'
        self.day_label = Label(self.learning_frame, text=day, font=('Helvetica', 18, "bold"))
        ##
        self.differ_button = Button(self.learning_frame, text='다른 유형')

        ##
        self.day_label.pack(padx=85, pady=10)
        self.differ_button.pack(padx=105, pady=10)
        ####
        df_Usermatrix = self.makeMatrix()
        df_Usermatrix = df_Usermatrix.where(df_Usermatrix['수준'] == LEVEL)
        df_Usermatrix = df_Usermatrix.dropna(axis=0)
        print(df_Usermatrix)
        while True:
            if LEVEL == '상':
                idx = random.randint(0, 0 + len(df_Usermatrix) - 1)
            elif LEVEL == '중':
                idx = random.randint(60, 60 + len(df_Usermatrix) - 1)
            else:
                idx = random.randint(120, 120 + len(df_Usermatrix) - 1)
            idx = df_Usermatrix['문제 번호'][idx]

            df_pivot_Usermatrix = df_Usermatrix.pivot_table('맞춤법 선호도', index='유저 아이디', columns='문제 번호').fillna(0)

            # 개인 맞춤형 아닌 , 특정 문제와 비슷한 문제 추천
            matrixUser = df_pivot_Usermatrix.values.T
            print(matrixUser.shape)

            """idx를 do-while vs SVD만 do-while"""
            """idx를 do-while"""

            SVD = TruncatedSVD(n_components=2)
            matrix = SVD.fit_transform(matrixUser)
            print(matrix.shape)
            print(matrix[0])

            corr = np.corrcoef(matrix)
            print(corr.shape)

            RecomP = df_pivot_Usermatrix.columns
            RecomP_list = list(RecomP)

            coffey_hands = RecomP_list.index(idx)

            corr_coffey_hands = corr[coffey_hands]

            df = pd.DataFrame(list(RecomP[(corr_coffey_hands > 0.8)]))
            df.columns = ['문제 번호']  # 칼럼명 '문제 번호'로 변경
            df = df.merge(problem_df, on='문제 번호')
            if len(df) >= 15: break
        df = df.loc[0:14]

        self.Save_User(df)

        self.problem()

    global SV
    SV = ""

    def Save_User(self, df):
        User_info = pd.read_excel('사용자 정보2.xlsx')
        writer = pd.ExcelWriter('recommended.xlsx', engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()

        recommendedProblem = pd.read_excel('recommended.xlsx')
        ###########################################################
        for a in recommendedProblem['문제 번호']:
            newbie = {
                '유저 아이디': [13],
                '문제 번호': [a],
                '수준': [LEVEL],
                '맞춤법 선호도': [5],
                '띄어쓰기 선호도': [1]
            }

            newbie = pd.DataFrame(newbie)
            #
            User_info = pd.concat([User_info, newbie])

        writer = pd.ExcelWriter('사용자 정보2.xlsx', engine='xlsxwriter')
        User_info.to_excel(writer, sheet_name='Sheet1', index=False)
        writer.close()
        return

    def analyzeTaste(self, day):
        global LEVEL
        # LEVEL = '하'  ###
        self.learning_frame = Frame(window, bg='pink')
        self.learning_frame.pack(fill='x')
        day = str(self.day) + '일'
        self.day_label = Label(self.learning_frame, text=day, font=('Helvetica', 18, "bold"))
        ##
        self.differ_button = Button(self.learning_frame, text='다른 유형')

        ##
        self.day_label.pack(padx=85, pady=10)
        self.differ_button.pack(padx=105, pady=10)
        ####
        ######## ########
        df_Usermatrix = self.makeMatrix()
        print(df_Usermatrix)
        # df_Usermatrix = df_Usermatrix.where(df_Usermatrix['수준'] == LEVEL)
        # df_Usermatrix = df_Usermatrix.dropna(axis=0)

        #####
        df_pivot_Usermatrix = df_Usermatrix.pivot_table('맞춤법 선호도', index='유저 아이디', columns='문제 번호').fillna(0)
        matrix = df_pivot_Usermatrix.values

        user_ratings_mean = np.mean(matrix, axis=1)
        matrix_user_mean = matrix - user_ratings_mean.reshape(-1, 1)

        U, sigma, Vt = svds(matrix_user_mean, k=2)
        sigma = np.diag(sigma)

        svd_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

        df_svd_preds = pd.DataFrame(svd_user_predicted_ratings, columns=df_pivot_Usermatrix.columns)

        already_rated, predictions = self.recommend_Problem(df_svd_preds, 13, problem_df, df_Usermatrix, 7)
        # predictions = predictions[predictions['Predictions'] > 0.8]
        predictions = predictions.reset_index()
        print(predictions)

        self.Save_User(predictions)
        self.problem()
        return

    def problem(self):
        recommended_df = pd.read_excel('recommended.xlsx')
        super().problem(self.n, recommended_df)

    def progress_bar(self):
        self.differ_button.destroy()  ##
        recommended_df = pd.read_excel('recommended.xlsx')
        super().progress_bar(recommended_df)

    def next_Q(self):
        super().next_Q()

    def marking(self, problem_type):

        super().marking(problem_type)

        # 돌아가기 만들기
        # self.differ_button.destroy()  ##
        OK_button = Button(self.problem_frame, text='확인', command=self.back_main)
        OK_button.pack()

    def back_main(self):
        self.problem_frame.destroy()
        # self.day_label.destroy()
        self.differ_button.destroy() ##
        self.learning_frame.destroy()

        lm = LearningManager()
        lm.selectMode()

    def recommend_Problem(self, df_svd_preds, user_id, ori_problem_df, ori_User_info, num_recommendations=7):
        global LEVEL

        user_row_number = user_id - 1
        sorted_user_predictions = df_svd_preds.iloc[user_row_number].sort_values(ascending=False)
        user_data = ori_User_info.where(ori_User_info['유저 아이디'] == user_id)
        user_data = user_data.dropna(axis=0)
        user_history = user_data.merge(ori_problem_df, on='문제 번호').sort_values(['맞춤법 선호도'], ascending=False)

        # 유저가 푼 문제 제외한 데이터를 추출
        recommendations = ori_problem_df[~ori_problem_df['문제 번호'].isin(user_history['문제 번호'])]
        recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), on='문제 번호')
        recommendations = recommendations.where(recommendations['수준'] == LEVEL)
        recommendations = recommendations.rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions',
                                                                                                       ascending=False).iloc[:num_recommendations, :]

        return user_history, recommendations

if __name__ == "__main__":
    LV = Level_Test(0)
    LV.level_test()
    window.mainloop()
    # LM = LearningManager()
    # LM.selectMode()
    # window.mainloop()