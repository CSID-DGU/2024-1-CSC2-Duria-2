from flask import Flask, request, jsonify
import openai
import os
import time
from datetime import datetime
from statistics import mean
from collections import Counter

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # JSON 응답에서 한글이 깨지지 않도록 설정

# OpenAI API 키 설정
def load_api_key():
    api_key = os.getenv('OPENAI_API_KEY')  # 환경 변수에서 API 키 가져오기
    if not api_key:
        api_key_file = './local.properties'  # 파일 경로 지정
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('OPENAI_API_KEY='):
                        api_key = line.split('=', 1)[1]
                        break
        except FileNotFoundError:
            print(f"Error: '{api_key_file}' 파일을 찾을 수 없습니다.")
        except Exception as e:
            print(f"Error while reading '{api_key_file}': {e}")
    return api_key

openai.api_key = load_api_key()

if openai.api_key is None:
    raise ValueError("OpenAI API 키를 로드할 수 없습니다. 환경 변수 또는 './local.properties' 파일을 확인하세요.")

# 모델 엔진 설정
model_engine = "gpt-4o"  # 또는 사용 가능한 다른 GPT-4 모델 이름으로 설정

# 대화 상태 클래스 정의
class DialogueState:
    def __init__(self, max_slots=3): # 실험을 위해 5->3
        self.intent = None
        self.slots = {}  # 메인 슬롯
        self.temp_slots = {}  # 임시 슬롯
        self.emotion = None
        self.proactivity = None
        self.consistency = True
        self.history = []
        self.max_slots = max_slots

    def update_intent_and_slots(self, intent, slots, dialogue_count, model_engine):
        self.intent = intent

        if dialogue_count <= 5:
            for key, value in slots.items():
                self.add_to_main_slots(key, value)
        else:
            for key, value in slots.items():
                if key in self.slots:
                    similarity = calculate_similarity(self.slots[key], value, model_engine)
                    if similarity >= 0.8:
                        self.add_to_main_slots(key, value)
                    else:
                        self.temp_slots[key] = value
                else:
                    self.add_to_main_slots(key, value)

    def add_to_main_slots(self, key, value):
        if len(self.slots) >= self.max_slots:
            oldest_key = next(iter(self.slots))
            del self.slots[oldest_key]

        self.slots[key] = value

# 전역 변수 초기화
dialogue_state = DialogueState()
response_times = []
dialogue_count = 0  # 대화 횟수 초기화

@app.route('/chat', methods=['POST'])
def chat():
    global dialogue_state, response_times, dialogue_count

    data = request.json
    user_input = data.get('user_input')

    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # 타임스탬프와 응답 시간 측정
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prompt_time = time.time()

    dialogue_count += 1  # 대화 횟수 증가

    # 현재 감정 분석
    current_emotion = analyze_emotion(user_input)
    dialogue_state.emotion = current_emotion

    # 의도와 슬롯 추출
    intent, slots = extract_intent_and_slots_with_similarity(user_input)
    dialogue_state.update_intent_and_slots(intent, slots, dialogue_count, model_engine)

    # 대화 이력 업데이트
    dialogue_state.history.append({
        'timestamp': timestamp,
        'user_input': user_input,
        'emotion': current_emotion,
        'intent': intent,
        'slots': slots
    })

    # 최대 최근 5개의 대화만 유지
    if len(dialogue_state.history) > 5:
        dialogue_state.history.pop(0)

    # 첫 5번의 대화에서는 적극성과 일관성을 판단하지 않음
    if dialogue_count <= 5:
        response = generate_response_first_phase(user_input, dialogue_state)
        proactivity = None
        consistency = None
    else:
        # 이후 대화에서는 적극성과 일관성을 고려하여 응답 생성
        consistency = check_consistency(dialogue_state)
        dialogue_state.consistency = consistency

        # 감정 변화 판단
        emotions = [turn['emotion'] for turn in dialogue_state.history[:-1]]  # 현재 감정 제외
        dominant_emotion = most_common(emotions) if emotions else None
        emotion_change = current_emotion != dominant_emotion

        # 평균 응답 시간 계산
        if len(response_times) > 1:
            average_response_time = mean(response_times[:-1])  # 현재 응답 시간 제외
        else:
            average_response_time = time.time() - prompt_time

        # 응답 시간 계산
        response_time = time.time() - prompt_time

        # 적극성 점수 계산
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

        response = generate_response_second_phase(user_input, dialogue_state)

        # 응답 시간을 리스트에 추가
        response_times.append(response_time)

    dialogue_state.history[-1]['response'] = response

    # 상태 정보 포함하여 응답 데이터 생성
    response_data = {
        'timestamp': timestamp,
        'response': response,
        'dialogue_state': {
            'intent': dialogue_state.intent,
            'slots': dialogue_state.slots,
            'temp_slots': dialogue_state.temp_slots,
            'emotion': dialogue_state.emotion,
            'proactivity': dialogue_state.proactivity,
            'consistency': dialogue_state.consistency,
            'history': dialogue_state.history[-5:]  # 최근 5개 대화만 포함
        }
    }

    return jsonify(response_data), 200

def analyze_emotion(text):
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

def extract_intent_and_slots_with_similarity(user_input):
    intent = determine_intent_and_slots(user_input)

    slots_prompt = f"""
    다음 문장에서 사용자의 슬롯 정보를 추출해주세요.
    문장: "{user_input}"
    슬롯은 아래 형식으로 추출해주세요.
    Slots:
    [슬롯명1]: [값1]
    [슬롯명2]: [값2]
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

def determine_intent_and_slots(user_input):
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

    total_checks += 1
    if current_intent in intent_counter:
        match_count += 1

    total_checks += 1
    if current_emotion in emotion_counter:
        match_count += 1

    previous_slots = {}
    for slots in slots_list:
        for key, value in slots.items():
            previous_slots.setdefault(key, set()).add(value)

    total_checks += len(current_slots)
    for key, value in current_slots.items():
        if key in previous_slots and value in previous_slots[key]:
            match_count += 1

    consistency_ratio = match_count / total_checks if total_checks > 0 else 0
    threshold = 0.5
    return consistency_ratio >= threshold

def generate_response_first_phase(user_input, dialogue_state):
    prompt = f"""
    당신은 친절하고 공감 능력이 뛰어난 AI 챗봇입니다.
    사용자: "{user_input}"
    """
    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        assistant_reply = response.choices[0].message.content.strip()
        return assistant_reply
    except openai.error.OpenAIError as e:
        print(f"Error during response generation: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."

def generate_response_second_phase(user_input, dialogue_state):
    proactivity_flag = dialogue_state.proactivity == "높음"
    consistency_flag = dialogue_state.consistency

    if proactivity_flag and consistency_flag:
        prompt = f"""
        당신은 활기차고 공감 능력이 뛰어난 AI 챗봇입니다.
        사용자: "{user_input}"
        """
    elif proactivity_flag and not consistency_flag:
        prompt = f"""
        당신은 활기차고 공감 능력이 뛰어난 AI 챗봇입니다.
        대화 내용에서 약간의 차이를 발견했어요. 혹시 이전에 말씀하신 내용과 현재 말씀하신 내용 중 어떤 것이 정확한지 알려주실 수 있을까요?
        사용자: '{user_input}'
        """
    elif not proactivity_flag and consistency_flag:
        prompt = f"""
        당신은 차분하고 사려 깊은 AI 챗봇입니다.
        적극성이 조금 부족한 것 같아요. 더 자세히 말씀해주실 수 있나요?
        사용자: '{user_input}'
        """
    else:
        prompt = f"""
        당신은 차분하고 사려 깊은 AI 챗봇입니다.
        이번에는 새로운 주제로 이야기해볼까요?
        사용자: '{user_input}'
        """

    try:
        response = openai.chat.completions.create(
            model=model_engine,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
        )
        assistant_reply = response.choices[0].message.content.strip()
        return assistant_reply
    except openai.error.OpenAIError as e:
        print(f"Error during response generation: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)