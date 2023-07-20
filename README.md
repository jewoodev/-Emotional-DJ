# Music resting place
일상에 지친 마음을 위로받고 쉬어갈 수 있게 하는 음악 상담사를 인공지능 기술을 이용해 구현하는 프로젝트   

# 프로젝트 기간
2023년 3월 3일 ~ 2023년 3월 9일

# 팀 구성원 및 역할
`우상욱`: Frontend, Backend, Chat Bot develop, ML/DL modeling, Data refinement & manifacturing    
`황도희`: Fill out the Chat Bot questionnaire, Data scraping(Youtube link), ML modeling  
`민병창`: DL modeling, subword tokenize, Algorithm of song's recommend   
`서영호`: Data preprocessing, ML modeling  
`신제우`: KoBERT modeling, Data scraping(lyrics)   

# 기술 스택
### Programming Language
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=black">

### ✔️Frond-end ✔️Back-end
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=black">

### Modeling
<img src="https://img.shields.io/badge/GridSearchCV-994B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/RandomSearchCV-184B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/Catboost-774B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/LSTM-664B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/GRU-593B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/RandomForest-444B4B?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/KoBERT-92AB4B?style=for-the-badge&logo=&logoColor=black">

### Data handling
<img src="https://img.shields.io/badge/Pandas-0A0A20?style=for-the-badge&logo=&logoColor=black"><img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=black">

### Database

<img src="https://img.shields.io/badge/PostgreSQL-302683?style=for-the-badge&logo=postgresql&logoColor=white">

# PREVIEW

<details>
<summary>PREVIEW 펼치기</summary>
<div markdown="1">

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/e63ff21e-5bb2-43fd-aac0-1de91c487885" alt="image" style="zoom: 50%;" />

> 시스템 아키텍처

'Music resting place'는 누군가에게 이야기하기 어려운 일이 있거나 나의 기쁜 일을 공감해줬으면 할 때에 대화를 나눌 수 있으며 그 대화를 통해 사용자의 감정에 알맞은 음악을 들려주어 위로와 공감을 받는 경험을 제공하는 웹 서비스입니다. 

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/475b3f31-80c7-4a2c-81dc-7f6a9892bdff" alt="image" style="zoom:50%;" />

> 사용자와 'Music resting place'의 챗봇이 대화하는 모습

대화를 나누다가 AI가 음악을 추천할 수 있을 만큼 대화가 오고 갔을 때 게이지가 가득 차며 내 감정이 어떤지 분석한 결과를 확인하는 것과 음악을 추천받을 수 있는 버튼이 활성화됩니다. 

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/b80d97df-1a7e-49d1-be41-1141143e9328" alt="image" style="zoom:50%;" />

> 감정 분석 결과를 보여주는 piechart

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/e260284f-e373-42dd-862d-959d329ca331" alt="image" style="zoom:50%;" />

> 음악 추천 카테고리

</div>
</details>



# 개발 관련 설명

<details>
<summary>개발 관련 설명 펼치기</summary>
<div markdown="1">
### 챗봇

사용자의 대화에 자연스럽게 응답할 수 있도록 사전학습 모델에 코사인 유사도를 이용해 챗봇을 구현하였습니다.

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/77835eb7-9408-4ea7-be09-835fc8e2f2e8" alt="image" style="zoom: 33%;" />

사전학습 모델로는 KO-ROBERTA 와 KO-BERT 두 가지 모델의 후보군에서 KO-ROBERTA를 선택했습니다. 선택 이유는 아래와 같습니다.

1. KO-BERT와 달리 dynamic masking을 사용
   - 크기가 큰 데이터 static masking을 적용하면 비효율적. 또 dynamic masking을 통해 성능 개선을 볼 수 있음.
2. 더 큰 batch size, byte 단위 level BPE를 통해 unknown 토큰 없이도, 적당한 크기의 서브워드 사전 학습을 진행할 수 있음
3. 실제 챗봇 구현 시 조금 더 자연스러운 응답을 선택하는 모델이었음

### 감정 분석 모델링

2가지의 전처리 방식을 활용하여 가장 스코어가 높은 모델을 찾았습니다.

1. 전처리 방식

   - Okt 기반 불용어처리 및 품사추출 및 구축 단어 사전 원핫인코딩

   - Sentence Piece tokenizer 기반 토크나이징 데이터(vocab은 kobert 사전 활용)

2. 모델링

   - ML : CATBOOST, XGBOSOT, LGBM, GBM 등
   - DL : LSTM, GRU, RNN, KO-BERT

<img src="https://github.com/jewoodev/music_resting_place/assets/105477856/627c7027-7abc-43c2-970f-8676bc957c7d" alt="image" style="zoom:50%;" />

OKT 기반 전처리 데이터를 활용했을 때, 일반적으로 평균 모델 성능이 좋았습니다.(KOBERT 제외)

### 음악 추천 알고리즘

![image](https://github.com/jewoodev/music_resting_place/assets/105477856/7ba93eb1-2106-41eb-8df0-fa21074a28b6)

1. 감성분석 모델을 통해 도출된 PROBA 값을 활용하여, 각 감정별로 0 ~ 1까지의 값을 가진 배열을 생성합니다. 이는 음악 데이터에도 적용되며, 챗봇을 통해 나온 데이터에도 적용됩니다.

   - 기존 크롤링되어 정제된 음악 가사 데이터 약 200건

   - 챗봇을 통해 나온 사용자의 텍스트

2. 사용자의 텍스트 감정분석 데이터와 음악 가사 데이터의 감정분석 데이터를 활용하여 추천을 진행합니다.

   - 감정이 비슷한 음악

     - 모든 음악과 사용자의 텍스트 감정분석 데이터의 코사인 유사도를 구합니다.

     - 음악의 다양성을 위해 모든 음악에 코사인 유사도 값에 랜덤으로 특정 값을 더해줍니다.

     - 코사인 유사도가 가장 큰 값을 선택하여 해당 음악을 추천합니다.

   - 감정이 반대인 음악

     - 사용자의 텍스트 감정 분석 값이 기쁨이 가장 높을 경우
       - 기쁨 값 중 절반을 가져와서, 남은 감정의 모든 데이터에 해당 값을 분포에 맞게 뿌려줍니다.

     - 사용자의 텍스트 감정 분석 값이 기쁨이 가장 높지 않은 경우
       - 기쁨을 제외한 나머지 감정의 절반을 전부 가져와서, 기쁨에 더해줍니다.

     - 음악의 다양성을 위해 모든 음악에 코사인 유사도 값에 랜덤으로 특정 값을 더해줍니다.

     - 코사인 유사도가 가장 큰 값을 선택하여 해당 음악을 추천합니다.

### 데이터베이스 활용

-  데이터베이스에 감정분석된 데이터, 추천된 노래, 그리고 유저의 좋아요/싫어요를 DB에 저장하고 이 데이터로 비슷한 감정을 가진 유저들의 정보를 기반으로 좋아요가 많이 찍힌 음악 위주로 추천하는 로직 구성

</div>

</details>

# References

## 참고 문헌

- SDSN 2022 세계 행복 지수  
- Bert와 gpt의 차이: scatter lab tech :https://tech.scatterlab.co.kr/transformer-review
- 챗봇 이미지 : https://www.flaticon.com/kr/free-icon/chatbot_2040946
- 멜론 : https://www.melon.com/
- 지니 : https://www.genie.co.kr/
- YouTubeMusic : https://music.youtube.com

## 데이터 출처 

AI-Hub : 웰니스 심리 상담 데이터셋, 감성분석 말뭉치