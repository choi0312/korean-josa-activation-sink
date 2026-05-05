from pathlib import Path
from typing import List

def build_default_corpus() -> List[str]:
    minimal_pairs = [
        "철수가 밥을 먹었다.", "철수는 밥을 먹었다.", "영희가 책을 읽었다.", "영희는 책을 읽었다.",
        "민수는 학교에 갔다.", "민수는 학교에서 공부했다.", "지민이 연구를 시작했다.", "지민은 연구를 시작했다.",
        "학생이 질문을 했다.", "학생은 질문을 했다.", "교수가 결과를 설명했다.", "교수는 결과를 설명했다.",
        "연구원이 반도체 공정을 분석했다.", "연구원은 반도체 공정을 분석했다.",
        "모델이 문맥을 이해했다.", "모델은 문맥을 이해했다.", "시스템이 오류를 기록했다.", "시스템은 오류를 기록했다.",
        "책을 읽었다.", "책도 읽었다.", "책만 읽었다.", "학교에 갔다.", "학교에서 공부했다.",
        "회의에 참석했다.", "회의에서 발표했다.", "데이터를 저장했다.", "데이터도 저장했다.", "데이터만 저장했다.",
        "나는 오늘 연구실에서 실험을 다시 실행했다.", "나는 오늘 연구실의 실험을 다시 실행했다.",
    ]
    natural_sentences = [
        "오늘 아침에 연구실에서 새 실험 결과를 확인했다.",
        "반도체 공정의 미세한 차이가 전체 수율에 큰 영향을 준다.",
        "이 모델은 긴 문맥을 유지할 때 불필요한 정보도 함께 끌고 가는 경향이 있다.",
        "한국어 문장에서 조사는 문법적 관계를 드러내는 중요한 표지다.",
        "교수님은 다음 주 세미나에서 관련 논문을 소개할 예정이다.",
        "학생들은 수업 후에 조별 과제 방향을 다시 정리했다.",
        "실험 로그에는 예상보다 큰 activation 값이 여러 번 나타났다.",
        "양자화 이후에 일부 문장의 생성 품질이 눈에 띄게 떨어졌다.",
        "우리는 형태소 분석 결과와 서브워드 토큰을 정렬해서 비교했다.",
        "모델 내부의 특정 채널은 짧은 기능어에서 유난히 크게 반응했다.",
        "연구팀은 이상치 채널이 특정 층에서 집중된다는 사실을 발견했다.",
        "이 결과가 단순한 위치 효과인지 확인하기 위해 추가 통제를 넣었다.",
        "한국어의 조사 체계는 영어의 전치사와는 다른 방식으로 구조를 표시한다.",
        "시퀀스 길이가 길어질수록 문맥을 보존하는 부담도 함께 커질 수 있다.",
        "토큰별 hidden state를 보면 몇몇 위치에서 값이 유난히 튀는 현상이 보인다.",
        "같은 명사라도 붙는 조사에 따라 모델의 내부 반응이 달라질 수 있다.",
        "문장 앞부분의 특수 토큰은 종종 attention sink처럼 작동하기도 한다.",
        "실험 설계가 단단해야 나중에 register token 개입도 설득력을 가진다.",
        "형태소 정렬이 틀리면 조사 효과처럼 보이는 착시가 생길 수 있다.",
        "조사와 어미는 둘 다 기능어지만 모델 내부에서는 다르게 작동할 가능성이 있다.",
        "이 실험은 한국어의 구조적 특성과 모델 효율화 문제를 함께 다룬다.",
        "우리는 조사 주변 토큰을 확대해서 local heatmap도 함께 제시했다.",
    ]
    subjects = ["철수", "영희", "민수", "지민", "연구원", "학생", "교수", "모델", "시스템", "분석가", "개발자"]
    subj_particles = ["이", "가", "은", "는"]
    objects = ["책", "논문", "데이터", "결과", "질문", "반도체 공정", "모델 출력", "로그", "보고서", "코드"]
    obj_particles = ["을", "를", "도", "만"]
    locs = ["학교", "연구실", "회의실", "서버", "데이터베이스", "실험 환경"]
    loc_particles = ["에", "에서", "으로", "부터", "까지"]
    verbs = ["읽었다", "분석했다", "저장했다", "설명했다", "확인했다", "기록했다", "정리했다", "비교했다"]
    generated = []
    for s in subjects:
        for sp in subj_particles:
            for o in objects:
                for op in obj_particles[:2]:
                    for v in verbs[:4]:
                        generated.append(f"{s}{sp} {o}{op} {v}.")
                        if len(generated) >= 140: break
                    if len(generated) >= 140: break
                if len(generated) >= 140: break
            if len(generated) >= 140: break
        if len(generated) >= 140: break
    generated_loc = []
    for s in subjects[:8]:
        for sp in subj_particles:
            for loc in locs:
                for lp in loc_particles:
                    generated_loc.append(f"{s}{sp} {loc}{lp} 결과를 확인했다.")
                    if len(generated_loc) >= 80: break
                if len(generated_loc) >= 80: break
            if len(generated_loc) >= 80: break
        if len(generated_loc) >= 80: break
    corpus = minimal_pairs + natural_sentences + generated + generated_loc
    return list(dict.fromkeys([x.strip() for x in corpus if x.strip()]))

def load_or_build_corpus(optional_corpus_txt: str, max_sentences: int) -> List[str]:
    p = Path(optional_corpus_txt)
    if p.exists():
        corpus = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        corpus = build_default_corpus()
    return list(dict.fromkeys(corpus))[:max_sentences]
