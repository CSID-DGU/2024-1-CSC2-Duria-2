import os
import openai
import time
from datetime import datetime
from statistics import mean
from collections import Counter

def main():
    print("DEBUG: main() 함수 시작")

    # OpenAI API 키 설정
    openai.api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기

    if not openai.api_key:
        # local.properties에서 API 키를 가져오기
        if os.path.exists('local.properties'):
            with open('local.properties', 'r') as f:
                for line in f:
                    if 'OPENAI_API_KEY=' in line:
                        openai.api_key = line.strip().split('=')[1]
                        break
        else:
            print("Error: local.properties 파일을 찾을 수 없습니다.")
            return

    if not openai.api_key:
        print("Error: OpenAI API 키가 설정되지 않았습니다. 직접 키를 설정하거나 환경 변수를 확인하세요.")
        return

    model_engine = "gpt-4o"
    print(f"{model_engine} 챗봇에 오신 것을 환영합니다! 종료하려면 'exit'를 입력하세요.\n")

    response_times = []
    dialogue_state = DialogueState(max_slots=3)  # 메인 슬롯 최대 개수를 10개로 설정 # 실험 위해서 3개로 설정
    dialogue_count = 0

    while True:
        try:
            prompt_time = time.time()
            user_input = input("You: ")  # 사용자 입력 대기
            response_time = time.time() - prompt_time

            if user_input.lower() in ['exit', 'quit', '종료']:
                print("챗봇을 종료합니다. 이용해 주셔서 감사합니다.")
                break

            dialogue_count += 1

            # 감정 분석
            current_emotion = analyze_emotion(user_input, model_engine)
            dialogue_state.emotion = current_emotion

            # 의도 및 슬롯 추출
            intent, slots = extract_intent_and_slots_with_similarity(user_input, model_engine)
            dialogue_state.update_intent_and_slots(intent, slots, dialogue_count, model_engine)

            # 대화 이력 업데이트
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dialogue_state.history.append({
                'timestamp': timestamp,
                'user_input': user_input,
                'emotion': current_emotion,
                'intent': intent,
                'slots': slots
            })

            # 대화 이력이 초과되면 제거
            if len(dialogue_state.history) > 5:
                dialogue_state.history.pop(0)

            # 응답 생성
            if dialogue_count <= 5:
                response = generate_response_first_phase(user_input, model_engine, dialogue_state)
                proactivity = None
                consistency = None
            else:
                # 일관성 및 적극성 계산
                consistency = check_consistency(dialogue_state)
                dialogue_state.consistency = consistency

                emotions = [turn['emotion'] for turn in dialogue_state.history[:-1]]
                dominant_emotion = most_common(emotions) if emotions else None
                emotion_change = current_emotion != dominant_emotion

                average_response_time = mean(response_times[:-1]) if len(response_times) > 1 else response_time

                proactivity = calculate_proactivity(
                    dialogue_state,
                    user_input,
                    response_time,
                    average_response_time,
                    current_emotion,
                    emotion_change,
                    prompt_time
                )
                dialogue_state.proactivity = proactivity

                response = generate_response_second_phase(user_input, model_engine, dialogue_state)

            # 대화 이력에 응답 추가
            dialogue_state.history[-1]['response'] = response
            response_times.append(response_time)

            # 상태 출력
            print("\n[현재 DST 상태]")
            dialogue_state.print_state(dialogue_count)

            print("\n[AI의 응답]")
            print(f"Timestamp: {timestamp}")
            print(f"응답: {response}")
            print("\n" + "=" * 50 + "\n")

        except Exception as e:
            print(f"오류가 발생했습니다: {e}")
            break


class DialogueState:
    def __init__(self, max_slots=10):
        self.intent = None
        self.slots = {}       # 메인 슬롯
        self.temp_slots = {}  # 임시 슬롯
        self.emotion = None
        self.proactivity = None
        self.consistency = True
        self.history = []
        self.max_slots = max_slots
        self.inconsistency_count = 0  # 일관성 깨진 횟수

    def update_intent_and_slots(self, intent, slots, dialogue_count, model_engine):
        self.intent = intent

        # 각 문장당 하나의 슬롯만 저장
        for key, value in slots.items():
            if dialogue_count <= 5:
                self.add_to_main_slots(key, value)
            else:
                if key in self.slots:
                    similarity = calculate_similarity(self.slots[key], value, model_engine)
                    if similarity >= 0.8:
                        self.add_to_main_slots(key, value)
                        self.inconsistency_count = 0  # 일관성 유지
                    else:
                        self.add_to_temp_slots(key, value)
                        self.inconsistency_count += 1  # 일관성 깨짐
                else:
                    self.add_to_temp_slots(key, value)
                    self.inconsistency_count += 1  # 일관성 깨짐

        # 임시 슬롯에서 일관성 깨진 횟수가 3회 연속이면 메인 슬롯 교체 -> 실험을 위해 2/3으로 교체
        if self.inconsistency_count >= 2:
            self.replace_main_slots_with_temp()
            self.inconsistency_count = 0  # 카운트 리셋

    def add_to_main_slots(self, key, value):
        if len(self.slots) >= self.max_slots:
            # 오래된 슬롯 제거 (FIFO 방식)
            oldest_key = next(iter(self.slots))
            del self.slots[oldest_key]
        self.slots[key] = value

    def add_to_temp_slots(self, key, value):
        # 임시 슬롯은 한 문장당 하나의 슬롯만 저장
        self.temp_slots = {key: value}

    def replace_main_slots_with_temp(self):
        # 메인 슬롯을 초기화하고 임시 슬롯으로 대체
        print("메인 슬롯을 임시 슬롯으로 대체합니다.")
        self.slots = self.temp_slots.copy()
        self.temp_slots.clear()

    def print_state(self, dialogue_count):
        print(f"의도(Intent): {self.intent}")
        print(f"메인 슬롯(Slots): {self.slots if self.slots else '해당 없음'}")
        print(f"임시 슬롯(Temp Slots): {self.temp_slots if self.temp_slots else '해당 없음'}")
        print(f"감정(Emotion): {self.emotion}")
        if dialogue_count > 5:
            print(f"적극성(Proactivity): {self.proactivity}")
            print(f"일관성(Consistency): {'일치' if self.consistency else '불일치'}")
        print(f"대화 이력(History):")
        for idx, turn in enumerate(self.history, 1):
            print(f" {idx}. [{turn['timestamp']}] 사용자: {turn['user_input']} (감정: {turn['emotion']}, 의도: {turn['intent']})")
            if 'response' in turn:
                print(f"    AI: {turn['response']}")


def analyze_emotion(text, model_engine):
    prompt = f"""
    다음 문장의 감정을 분석해주세요.
    문장: "{text}"
    감정은 '긍정', '부정', '중립' 중 하나로 답변해주세요.
    """
    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        emotion = response.choices[0].message.content.strip()
        if "긍정" in emotion:
            return "긍정"
        elif "부정" in emotion:
            return "부정"
        else:
            return "중립"
    except openai.error.OpenAIError as e:
        print(f"Error during emotion analysis: {e}")
        return "알 수 없음"


def calculate_similarity(sentence1, sentence2, model_engine):
    prompt = f"""
    다음 두 문장이 의미적으로 얼마나 유사한지 0에서 1 사이의 점수로 평가해주세요.
    1은 완전히 같음을, 0은 완전히 다름을 의미합니다.

    문장 1: "{sentence1}"
    문장 2: "{sentence2}"
    점수만 반환해주세요.
    """
    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0,
        )
        similarity_score = float(response.choices[0].message.content.strip())
        return similarity_score
    except openai.error.OpenAIError as e:
        print(f"Error during similarity calculation: {e}")
        return 0.0


def extract_intent_and_slots_with_similarity(user_input, model_engine):
    intent = determine_intent_and_slots(user_input, model_engine)

    slots_prompt = f"""
    다음 문장에서 사용자의 의도와 슬롯 정보를 추출해주세요.
    문장: "{user_input}"
    """
    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=[{"role": "user", "content": slots_prompt}],
            max_tokens=150,
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        lines = content.split('\n')
        extracted_slots = {}
        for line in lines:
            if ':' in line and not line.startswith('Intent:'):
                slot_name, slot_value = line.split(':', 1)
                extracted_slots[slot_name.strip()] = slot_value.strip()

        return intent, extracted_slots
    except openai.error.OpenAIError as e:
        print(f"Error during intent and slot extraction: {e}")
        return None, {}


def determine_intent_and_slots(user_input, model_engine):
    standard_intents = {
        "추천": ["음식 추천 요청", "추천 요청", "메뉴 추천"],
        "주문": ["음식 주문", "배달 요청"],
        "위치 확인": ["현재 위치 요청", "주소 요청"]
    }
    best_intent = None
    best_score = 0.0

    for intent, examples in standard_intents.items():
        for example in examples:
            similarity = calculate_similarity(user_input, example, model_engine)
            if similarity > best_score:
                best_score = similarity
                best_intent = intent

    return best_intent if best_score >= 0.8 else "알 수 없음"


def calculate_proactivity(dialogue_state, user_input, response_time, average_response_time, current_emotion, emotion_change, prompt_time):
    LENGTH_THRESHOLD = 10  # 적절한 대화 길이 기준
    TIME_INTERVAL_THRESHOLD = 10  # 적절한 대화 간의 간격 (초)

    if len(dialogue_state.history) > 1:
        last_interaction_time = time.mktime(
            datetime.strptime(dialogue_state.history[-2]['timestamp'], "%Y-%m-%d %H:%M:%S").timetuple()
        )
        time_interval = prompt_time - last_interaction_time
    else:
        time_interval = None

    proactivity_score = 0

    if response_time < average_response_time:
        proactivity_score += 1
    else:
        proactivity_score -= 1

    if not emotion_change:
        proactivity_score += 1
    else:
        proactivity_score -= 1

    if len(user_input) >= LENGTH_THRESHOLD:
        proactivity_score += 1
    else:
        proactivity_score -= 1

    if time_interval is not None:
        if time_interval <= TIME_INTERVAL_THRESHOLD:
            proactivity_score += 1
        else:
            proactivity_score -= 1

    return "높음" if proactivity_score >= 1 else "낮음"


def most_common(lst):
    if lst:
        data = Counter(lst)
        return data.most_common(1)[0][0]
    else:
        return None


def check_consistency(dialogue_state):
    intents = [turn['intent'] for turn in dialogue_state.history[:-1]]
    emotions = [turn['emotion'] for turn in dialogue_state.history[:-1]]
    slots_list = [turn['slots'] for turn in dialogue_state.history[:-1]]

    intent_counter = Counter(intents)
    emotion_counter = Counter(emotions)

    current_intent = dialogue_state.history[-1]['intent']
    current_emotion = dialogue_state.history[-1]['emotion']
    current_slots = dialogue_state.history[-1]['slots']

    match_count = 0
    total_checks = 0

    # 의도 일치 여부 확인 (메인 슬롯에 가중치 부여)
    total_checks += 2
    if current_intent in intent_counter:
        match_count += 2  # 가중치 2배

    # 감정 일치 여부 확인
    total_checks += 1
    if current_emotion in emotion_counter:
        match_count += 1

    # 슬롯 일치 여부 확인 (메인 슬롯에 가중치 부여)
    total_checks += len(current_slots) * 2
    for key, value in current_slots.items():
        if key in dialogue_state.slots and dialogue_state.slots[key] == value:
            match_count += 2  # 메인 슬롯 일치 시 가중치 2배
        else:
            for slots in slots_list:
                if key in slots and slots[key] == value:
                    match_count += 1

    consistency_ratio = match_count / total_checks
    threshold = 0.6
    return consistency_ratio >= threshold


def generate_response_first_phase(user_input, model_engine, dialogue_state):
    messages = [{"role": "system", "content": "당신은 친절하고 공감 능력이 뛰어난 AI 챗봇입니다."}]

    # 대화 이력을 메시지에 추가
    for turn in dialogue_state.history:
        messages.append({"role": "user", "content": turn['user_input']})
        if 'response' in turn:
            messages.append({"role": "assistant", "content": turn['response']})

    # 현재 사용자 입력 추가
    messages.append({"role": "user", "content": user_input})

    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"응답 생성 중 오류 발생: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."


def generate_response_second_phase(user_input, model_engine, dialogue_state):
    messages = []

    # 시스템 메시지에 적극성 및 일관성 정보를 포함
    if dialogue_state.consistency:
        system_content = "당신은 적극적이고 일관성 있는 AI 챗봇입니다."
    else:
        system_content = "당신은 차분하고 신중한 AI 챗봇입니다."
    messages.append({"role": "system", "content": system_content})

    # 대화 이력을 메시지에 추가
    for turn in dialogue_state.history:
        messages.append({"role": "user", "content": turn['user_input']})
        if 'response' in turn:
            messages.append({"role": "assistant", "content": turn['response']})

    # 현재 사용자 입력 추가
    messages.append({"role": "user", "content": user_input})

    # 일관성이 깨지고 임시 슬롯이 메인 슬롯으로 대체된 경우 프롬프트 엔지니어링 적용
    if dialogue_state.inconsistency_count == 0 and dialogue_state.temp_slots:
        previous_slot_content = ', '.join(f"{k}: {v}" for k, v in dialogue_state.slots.items())
        temp_slot_content = ', '.join(f"{k}: {v}" for k, v in dialogue_state.temp_slots.items())
        assistant_prompt = f"이전에 {previous_slot_content}에 대해 이야기하셨는데, 지금은 {temp_slot_content}에 대해 말씀하셨네요! {temp_slot_content}가 맞을까요?"
        messages.append({"role": "assistant", "content": assistant_prompt})

    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except openai.error.OpenAIError as e:
        print(f"응답 생성 중 오류 발생: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."

if __name__ == "__main__":
    main()