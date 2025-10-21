# JobPlanet Review Text Mining  
잡플래닛 리뷰 기반 기업 평판 분석 프로젝트 (SAPI 팀)

---

## 🔍 목차  

<details>
<summary>목차 보기 (클릭)</summary>

1. [프로젝트 개요](#-프로젝트-개요)  
2. [기술 스택](#-기술-스택)  
3. [데이터 수집](#-데이터-수집)  
4. [데이터 전처리](#-데이터-전처리)  
5. [EDA 및 가설 검정](#-eda-및-가설-검정)  
6. [모델링](#-모델링)  
7. [결과 및 인사이트](#-결과-및-인사이트)  
8. [활용 방안 및 개선 아이디어](#-활용-방안-및-개선-아이디어)  
9. [향후 계획](#-향후-계획)  

</details>

---

## 📋 프로젝트 개요  

**목표)**  
- 잡플래닛 리뷰 데이터를 활용해 특정 기업(이스트소프트)의 **장점·단점·경영진 의견**을 텍스트 마이닝으로 분석  
- 실제 재직자들의 리뷰를 기반으로 기업문화 및 HR 개선 인사이트 도출  

**역할)**  
- 팀장으로서 데이터 수집·EDA·가설 검정·모델링 총괄  
- TF-IDF, 감성사전, KNN 기반 유사도 탐색 기법을 활용한 텍스트 분석 담당  

**성과)**  
- 367건의 리뷰를 정제·분석하여 5개 가설 검증  
- TF-IDF, 감성 분석, 키워드 유사도 모델링을 통한 **데이터 기반 기업문화 진단 체계 구축**  

---

## 🛠 기술 스택  

- **언어**: Python 3.8+  

- **웹 크롤링**:  
  - `Selenium` – 브라우저 자동 제어 및 JobPlanet 로그인/페이지 이동  
  - `BeautifulSoup (bs4)` – HTML 파싱 및 리뷰 데이터 추출  
  - `chromedriver`, `WebDriverWait`, `expected_conditions` – 크롬 드라이버 제어 및 요소 대기 처리  
  - `re`, `time`, `random`, `csv` – 텍스트 정제, 요청 지연, CSV 저장  

- **데이터 처리**:  
  - `pandas`, `numpy`, `scipy` – 데이터 프레임 관리, 통계 연산, 가설 검정  

- **자연어 처리**:  
  - `konlpy` (Okt 형태소 분석기), `re`, `collections` – 한글 형태소 분석 및 키워드 빈도 분석  

- **시각화**:  
  - `matplotlib`, `seaborn`, `wordcloud` – 단어 빈도, TF-IDF, 감성 분석 결과 시각화  

- **모델링**:  
  - `scikit-learn` – TF-IDF 벡터화, KNN 기반 유사도 탐색  

- **감성 분석**:  
  - [KNU 한국어 감성사전](https://github.com/park1200656/KnuSentiLex) – 문장 단위 감성 점수 산출  

- **환경**:  
  - `Jupyter Notebook` – 분석 코드 실행 및 시각화 환경  

---

## 📊 데이터 수집  

- **출처**: [JobPlanet - 이스트소프트 리뷰 페이지](https://www.jobplanet.co.kr/companies/58863/reviews/%EC%9D%B4%EC%8A%A4%ED%8A%B8%EC%86%8C%ED%94%84%ED%8A%B8)  
- **수집 방법**: Selenium을 이용한 리뷰 크롤링 자동화  

| 항목 | 내용 |
|------|------|
| **데이터 크기** | 367행 × 4열 |
| **컬럼 구성** | 제목, 장점, 단점, 경영진에게 하고 싶은 말 |
| **전처리 요약** | HTML 태그 제거, 결측치 처리, 불용어 제거, 형태소 단위 정제 |

---

## 🧹 데이터 전처리  

1. **텍스트 정제**
   - HTML 태그, 특수문자, 공백 제거  
   - 불용어 제거(`stopwords`) 및 소문자 통일  
   ```python
   df['장점'] = df['장점'].str.replace('[^가-힣 ]', '', regex=True)
   df['단점'] = df['단점'].str.replace('[^가-힣 ]', '', regex=True)
   ```

2. **형태소 분석**
   - `Okt` 기반 명사 추출 → 텍스트 기반 키워드 분석  
   ```python
   from konlpy.tag import Okt
   okt = Okt()
   df['pros_nouns'] = df['장점'].apply(lambda x: okt.nouns(x))
   ```

3. **결측치 제거**
   - `NaN` 값 제거 후 인덱스 재정렬  
   ```python
   df.dropna(inplace=True)
   df.reset_index(drop=True, inplace=True)
   ```

---

## 📈 EDA 및 가설 검정  

<img width="983" height="413" alt="image" src="https://github.com/user-attachments/assets/8bea7ddc-e789-4f14-8ec8-22ccafee2aa8" />

<EDA (1) - 장점/단점 빈도 TOP 10>

<img width="239" height="244" alt="image" src="https://github.com/user-attachments/assets/382505db-abd4-4d69-b37a-9121e5dc51d1" />

> 공통적으로 **분위기**, **회사**, **사람**, **업무**, **개발자**가 많이 언급되었다.  
> 이 중에서 **분위기**와 **업무**는 장점 쪽에서 더 자주 등장하여 **긍정 요인**으로 분류되었고,  
> **사람**은 단점 쪽에서 더 많이 등장하여 **부정 요인**으로 분류되었다.  
>  
> 또한 **회사**와 **개발자**는 긍·부정 양쪽 모두에서 고르게 등장하여  
> 조직 내 다양한 인식 차이를 반영하는 **복합적 키워드**로 확인되었다.  
>  
> 장점 쪽에서는 **분위기**, **자유**, **연차**, **업무**, **문화**, **눈치**, **야근** 등이 자주 언급되었고,  
> 단점 쪽에서는 **사람**, **느낌**, **부서**, **성장**, **부족**, **직원** 등이 자주 등장하였다.  

<img width="574" height="459" alt="image" src="https://github.com/user-attachments/assets/61db7e22-455c-4ec6-a5e6-e8d6cf8bb6da" />

<EDA (2) - 제목 WordCloud>  
*(다른 WordCloud 결과는 생략하고 대표 이미지만 첨부)*

> WordCloud 결과, **‘회사’, ‘개발자’, ‘사람’, ‘기업’** 단어가 가장 크게 나타나  
> 리뷰 제목에서 **가장 자주 언급된 핵심 키워드**임을 확인할 수 있다.  
>  
> 그다음으로는 **‘분위기’, ‘성장’, ‘개발’, ‘자유’, ‘신입’** 등의 단어가 비교적 크게 나타났으며,  
> 이들이 리뷰 제목에서 **빈도 상위 그룹**에 포함되었다.  
>  
> 전반적으로 다양한 단어가 분포해있다.

<img width="633" height="717" alt="image" src="https://github.com/user-attachments/assets/13a48daa-d7ad-4f2c-8adb-fb490da52b31" />

<EDA (3) - 장점/단점 TF-IDF Top 10>

<img width="295" height="281" alt="image" src="https://github.com/user-attachments/assets/798ae6a7-b3e3-4ffb-90dd-cbcdd134d58d" />

> **빈도 분석**과 **TF-IDF 분석** 모두에서 공통적으로 비슷한 핵심 키워드가 도출되었다.  
> 그러나 두 분석 방식의 **순위에는 일부 차이**가 관찰되었는데,  
> 이는 **빈도 분석이 단순히 많이 언급된 단어를 강조**하는 반면,  
> **TF-IDF는 각 그룹(장점·단점)을 더 잘 구분해주는 특성 단어**에  
> 상대적으로 높은 가중치를 부여하기 때문이다.  
>
> 따라서 TF-IDF 결과는 리뷰 간의 **내용적 차별성을 드러내는 단어들**에  
> 더 초점을 맞춘 분석이라고 볼 수 있다. 



**가설 1️⃣**: “이스트소프트는 장점보다 단점이 더 많은 회사일 것이다.”  
<img width="974" height="647" alt="image" src="https://github.com/user-attachments/assets/253db8e7-f0f8-4355-8e53-7cca816621f7" />

- Welch’s t-test로 검정  
  ```python
  from scipy.stats import ttest_ind
  t_stat, p_value = ttest_ind(df['pros_len'], df['cons_len'], equal_var=False)
  ```
  - p-value > 0.05 → 귀무가설 채택  
  - 결론: 통계적으로 유의한 차이가 없음  
  - “이스트소프트는 장점보다 단점이 더 많은 회사라고 보기는 어렵다.”
  - 
**가설 2️⃣**: “신입 교육의 부족이 지원자 감소의 원인이다.”  
<img width="603" height="362" alt="image" src="https://github.com/user-attachments/assets/f3cca82b-7659-40fd-b9ae-fc17cbe1fbe4" />
 
- 대응표본 t-검정 (Paired t-test)으로 검정  
  ```python
  from scipy.stats import ttest_rel
  t_stat, p_value = ttest_rel(sentiment_df['긍정적 이미지'], sentiment_df['부정적 이미지'])
  ```
  - p-value > 0.05 → 귀무가설 채택
  - 결론: 신입 교육 관련 긍·부정 언급의 평균 차이는 통계적으로 유의하지 않음
  → “신입 교육 부족이 지원자 감소의 주요 원인이라 보기는 어렵다."

**가설 3️⃣**: “개발 부서의 업무 강도가 타 부서보다 높다.”  
- ‘개발’, ‘야근’, ‘업무량’ 키워드 빈도 분석 → 유의미한 차이 없음  

**가설 4️⃣**: “인재 유출의 원인은 성장 가능성 부재이다.”  
- ‘성장’, ‘커리어’, ‘비전’ 부정 감성 비율 높음 → 채택  

**가설 5️⃣**: “운영진에게 쓴소리가 칭찬보다 많다.”  
- 감성 점수 평균 비교 시 유의한 차이 없음 → 근거 미약  

---

## 🧠 모델링  

1. **TF-IDF 벡터화**
   ```python
   from sklearn.feature_extraction.text import TfidfVectorizer
   vectorizer = TfidfVectorizer(max_features=500)
   X = vectorizer.fit_transform(df['단점'])
   ```

2. **KNN 기반 유사도 탐색**
   ```python
   from sklearn.neighbors import NearestNeighbors
   model = NearestNeighbors(metric='cosine')
   model.fit(X)
   ```

3. **감성 분석 (KNU 사전 활용)**
   ```python
   from KnuSentiLex import KnuSL
   senti = KnuSL()
   score = senti.data['야근']  # {'word': '야근', 'polarity': -1}
   ```

4. **유사 리뷰 검색 예시**
   > “야근” 키워드와 관련된 리뷰를 유사도 순으로 출력  
   ```python
   query_vec = vectorizer.transform(['야근'])
   distances, indices = model.kneighbors(query_vec)
   ```

---

## 💡 결과 및 인사이트  

| 구분 | 주요 키워드 | 해석 |
|------|--------------|------|
| **장점** | 자유, 분위기, 연차, 문화, 업무 | 수평적이고 자유로운 기업문화 중심 |
| **단점** | 성장, 부족, 사람, 부서 | 성장 한계, 부서 간 불균형 존재 |
| **운영진 의견** | 칭찬 < 쓴소리 | 관리 체계와 의사소통에 대한 개선 요구 |

📈 **요약 인사이트**  
- “성장 가능성 부재”는 인재 유출의 주요 원인 중 하나로 확인됨  
- 직원들은 조직 분위기보다 **경영 구조적 개선**을 더 요구  
- 단순한 복지 강화보다 **직무 성장 경로 확립**이 핵심 과제  

---

## 🚀 활용 방안 및 개선 아이디어  

- **브랜딩 강화**: 긍정 키워드(‘복지’, ‘워라벨’) 중심 홍보 콘텐츠 제작  
- **조직 진단**: 부정 감성 키워드 기반 리스크 모니터링 시스템 구축  
- **이직 위험 예측**: 특정 부정 키워드 증가율을 기반으로 퇴사 위험도 평가  
- **대시보드 구축 제안**: Streamlit 기반 키워드별 감성 분석 시각화  

---

## 📌 향후 계획  

- **정교한 감성 분석 모델 도입**  
  - 감성사전 → BERT 기반 감정 분류기로 전환  
- **대시보드 시각화 고도화**  
  - 부서별·기간별 감성 트렌드 추적 기능 추가  
- **다중 기업 비교 분석 확장**  
  - NHN, 카카오 등 동종 기업 리뷰와의 교차 분석  
- **자동 보고서 생성**  
  - ChatGPT API 연동으로 기업 평판 리포트 자동화  

---

> **작성자**: 김정철 (팀장 / 데이터 수집·EDA·가설 검정·모델링 총괄)  
> **소속 팀**: SAPI (사피)  
> **데이터 출처**: [JobPlanet](https://www.jobplanet.co.kr/)
