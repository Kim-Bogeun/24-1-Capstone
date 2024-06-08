## Data 

### Selenium Crawling
각 리뷰와 리뷰에 해당하는 이미지들, 상품 meta data 및 review meta data 수집   
쿠팡 리뷰 특성 상 리뷰어 옆에 실명리뷰어, Top100 등의 태그가 붙어 있어 해당 정보도 수집   

쿠팡체험단 태그가 활성화된 경우 label : 1 , o.w. 0

### Data Description
분류를 위한 최종 사용 데이터
review_num	review_clean	Del	Category	flag	line_breaks

| Column    | Description    | 비고     |   
| ------- | ---: | ---    |
| label   | 체험단리뷰 여부  ||
| review  | 리뷰 원본   ||
| review_full  | 제목과 합친 리뷰 텍스트  ||
| review_clean  | cleaned 텍스트  |불용어 제거, 분류를 확실하게 할 수 있는 단어 및 문장 제거|
| helpfulness  | 도움이 되었어요 수 |   review meta|
| rate  | 별점  | review meta|
| realname_reviewer  | 실명리뷰어 여부  | reviewer meta|
| top_reviewer  | 탑100, top50등 탐 리뷰어 여부  |reviewer meta|
| title1  | 제목 유무  | review meta |
| review_num  | 글자 수  | review meta|
| line_breaks  | 줄바꿈 수  | review meta|
