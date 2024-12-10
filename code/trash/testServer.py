# from flask import Flask, request, jsonify, make_response
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# import torch
# import openai
# import os
# import json

# app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False  # 추가: JSON 응답에서 ASCII가 아닌 문자도 그대로 표시

# def load_api_key_from_local_properties(file_path):
#     api_key = None
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line.startswith('OPENAI_API_KEY='):
#                     api_key = line.split('=', 1)[1]
#                     break
#     except FileNotFoundError:
#         print(f"Error: '{file_path}' 파일을 찾을 수 없습니다.")
#     except Exception as e:
#         print(f"Error while reading '{file_path}': {e}")
#     return api_key

# # OpenAI API 설정
# api_key_file = './local.properties'  # 파일 경로 지정
# openai.api_key = load_api_key_from_local_properties(api_key_file)

# if openai.api_key is None:
#     raise ValueError("OpenAI API 키를 로드할 수 없습니다. './local.properties' 파일을 확인하세요.")


# # 감정 분석 모델 로드 (beomi/KcELECTRA-base-v2022)
# bert_model_name = "beomi/KcELECTRA-base-v2022"
# emotion_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
# emotion_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# # 대화 로그
# responses_log = []

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.json
#     instruction = data.get('instruction', '')

#     if not instruction:
#         return jsonify({"error": "No input provided"}), 400

#     # 타임스탬프 생성
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # 일관성 및 감정 분석 초기화
#     previous_log = None
#     consistency = 1  # 기본적으로 일관성 있다고 설정

#     if responses_log:
#         # 이전 대화와 유사성을 분석
#         previous_log, similarity_score, time_weight = select_most_similar_conversation(instruction, responses_log)
#         if previous_log:
#             previous_user_input = previous_log['instruction']
#             previous_emotion, current_emotion = check_emotion_and_negativity(
#                 previous_user_input, instruction, emotion_model, emotion_tokenizer)
#             emotion_change = abs(previous_emotion - current_emotion)
#             consistency_score = (similarity_score * 0.8) + (time_weight * 0.15) - (emotion_change * 0.05)

#             # 일관성 판단
#             if consistency_score < 0.7:
#                 consistency = 0
#     else:
#         current_emotion = None  # responses_log가 비어 있을 때 감정 초기화

#     # 응답 생성
#     response_text = generate_response(instruction, consistency)

#     # 대화 로그 저장
#     responses_log.append({'timestamp': timestamp, 'instruction': instruction, 'response': response_text})

#     # 최근 N개의 대화만 전송 (예: 최근 10개)
#     recent_conversations = responses_log[-10:]

#     # 응답 데이터 생성
#     response_data = {
#         'response': response_text,
#         'similar_conversation': previous_log['instruction'] if previous_log else None,
#         'consistency_score': consistency_score if previous_log else None,
#         'emotion': current_emotion if previous_log else None,
#         'conversations': recent_conversations  # 추가: 최근 대화 목록 포함
#     }

#     # JSON 직렬화 (ensure_ascii=False 설정)
#     response_json = json.dumps(response_data, ensure_ascii=False)

#     # 응답 생성
#     response = make_response(response_json)
#     response.headers['Content-Type'] = 'application/json; charset=utf-8'

#     return response, 200

# def select_most_similar_conversation(current_instruction, responses_log):
#     if not responses_log:
#         return None, 0, 0  # 유사한 대화가 없을 경우 일관성 0으로 반환

#     # 타임스탬프 가중치 계산
#     current_time = datetime.now()
#     time_weights = []
#     for log in responses_log:
#         time_diff = (current_time - datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M:%S")).total_seconds()
#         time_weights.append(max(1, 1 / (time_diff + 1)))

#     # 코사인 유사도 계산
#     vectorizer = TfidfVectorizer()
#     all_user_inputs = [log['instruction'] for log in responses_log] + [current_instruction]
#     tfidf_matrix = vectorizer.fit_transform(all_user_inputs)
#     cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

#     # 유사도와 타임스탬프 가중치를 곱해 최종 가중치를 계산
#     weighted_similarities = cosine_similarities * time_weights
#     most_similar_index = weighted_similarities.argmax()

#     # 가장 유사한 대화의 유사도 값 및 타임스탬프 가중치 반환
#     return responses_log[most_similar_index], cosine_similarities[most_similar_index], time_weights[most_similar_index]

# def check_emotion_and_negativity(previous_response, current_response, emotion_model, emotion_tokenizer):
#     # 빈 입력 처리
#     if not previous_response.strip() or not current_response.strip():
#         return 0, 0  # 기본값으로 감정 일치(0, 0) 처리

#     # 입력 텍스트 길이 제한 (예: 512 토큰)
#     max_length = 512
#     inputs = emotion_tokenizer(
#         [previous_response[:max_length], current_response[:max_length]],
#         return_tensors="pt",
#         padding=True,
#         truncation=True
#     )

#     with torch.no_grad():
#         outputs = emotion_model(**inputs)
#         logits = outputs.logits
#         predicted_class = logits.argmax(dim=-1)

#     previous_emotion = predicted_class[0].item()
#     current_emotion = predicted_class[1].item()

#     return previous_emotion, current_emotion

# def generate_response(instruction, consistency):
#     # 일관성이 있을 때와 없을 때의 프롬프트 생성
#     if consistency:
#         system_prompt = '''당신은 친근하고 유머러스한 AI 챗봇입니다. 사용자의 질문에 대해 적극적이고 자신감 있게 답변해야 합니다.
#         만약 사용자의 질문과 맥락이 일치하거나 일관성이 있다고 판단되면, 확신을 가지고 답변하세요.'''
#     else:
#         system_prompt = '''당신은 친절하고 사려 깊은 AI 챗봇입니다. 사용자의 질문에 대해 추가적인 정보를 요구하거나, 질문이 명확하지 않은 경우에는 재질문을 통해 확인을 하십시오.
#         사용자의 질문과 이전 대화가 일치하지 않는다고 판단되면, "조금 더 구체적으로 설명해 주실 수 있나요?" 또는 "다른 의미로 말씀하신 것인가요?"와 같이 확인 질문을 하세요.'''

#     try:
#         # OpenAI API 호출
#         response = openai.chat.completions.create(  # 이 부분에서 오류 자주 발생
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": instruction}
#             ],
#             max_tokens=150,
#             temperature=0.7,
#             top_p=0.9,
#             n=1,
#         )

#         # 응답에서 메시지 내용 추출
#         assistant_reply = response.choices[0].message.content.strip()
#         return assistant_reply

#     except openai.OpenAIError as e:
#         print(f"OpenAI API Error: {e}")
#         return f"OpenAI API Error: {e}"

#     except Exception as e:
#         print(f"Unhandled Exception: {e}")
#         return f"Unhandled Exception: {e}"


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

#=======================================================================

# from flask import Flask, request, jsonify
# import openai
# import os
# import time
# from datetime import datetime
# from statistics import mean
# from collections import Counter

# app = Flask(__name__)
# app.config['JSON_AS_ASCII'] = False  # JSON 응답에서 한글이 깨지지 않도록 설정

# # OpenAI API 키 설정
# def load_api_key_from_local_properties(file_path):
#     api_key = None
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if line.startswith('OPENAI_API_KEY='):
#                     api_key = line.split('=', 1)[1]
#                     break
#     except FileNotFoundError:
#         print(f"Error: '{file_path}' 파일을 찾을 수 없습니다.")
#     except Exception as e:
#         print(f"Error while reading '{file_path}': {e}")
#     return api_key

# api_key_file = './local.properties'  # 파일 경로 지정
# openai.api_key = load_api_key_from_local_properties(api_key_file)

# if openai.api_key is None:
#     raise ValueError("OpenAI API 키를 로드할 수 없습니다. './local.properties' 파일을 확인하세요.")

# # 대화 상태 클래스 정의
# class DialogueState:
#     def __init__(self):
#         self.intent = None
#         self.slots = {}
#         self.emotion = None
#         self.proactivity = None
#         self.consistency = True
#         self.history = []

#     def update_intent_and_slots(self, intent, slots):
#         self.intent = intent
#         self.slots.update(slots)

# # 전역 변수 초기화
# dialogue_state = DialogueState()
# response_times = []
# dialogue_count = 0  # 대화 횟수 초기화

# @app.route('/chat', methods=['POST'])
# def chat():
#     global dialogue_state, response_times, dialogue_count

#     data = request.json
#     user_input = data.get('user_input')

#     if not user_input:
#         return jsonify({"error": "No input provided"}), 400

#     # 타임스탬프와 응답 시간 측정
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     start_time = time.time()

#     dialogue_count += 1  # 대화 횟수 증가

#     # 현재 감정 분석
#     current_emotion = analyze_emotion(user_input, "gpt-3.5-turbo")
#     dialogue_state.emotion = current_emotion

#     # 의도와 슬롯 추출
#     intent, slots = extract_intent_and_slots(user_input, "gpt-3.5-turbo")
#     dialogue_state.update_intent_and_slots(intent, slots)

#     # 대화 이력 업데이트
#     dialogue_state.history.append({
#         'timestamp': timestamp,
#         'user_input': user_input,
#         'emotion': current_emotion,
#         'intent': intent,
#         'slots': slots
#     })

#     # 최대 최근 5개의 대화만 유지
#     if len(dialogue_state.history) > 5:
#         dialogue_state.history.pop(0)

#     # 첫 5번의 대화에서는 적극성과 일관성을 판단하지 않음
#     if dialogue_count <= 5:
#         response = generate_response_first_phase(user_input, "gpt-3.5-turbo", dialogue_state)
#         proactivity = None
#         consistency = None
#     else:
#         # 이후 대화에서는 적극성과 일관성을 고려하여 응답 생성
#         consistency = check_consistency(dialogue_state)
#         dialogue_state.consistency = consistency

#         # 감정 변화 판단
#         emotions = [turn['emotion'] for turn in dialogue_state.history[:-1]]  # 현재 감정 제외
#         if emotions:
#             dominant_emotion = most_common(emotions)
#             emotion_change = current_emotion != dominant_emotion
#         else:
#             emotion_change = False  # 초기 상태에서는 감정 변화 없음

#         # 평균 응답 시간 계산
#         if len(response_times) > 1:
#             average_response_time = mean(response_times[:-1])  # 현재 응답 시간 제외
#         else:
#             average_response_time = time.time() - start_time

#         # 응답 시간 계산
#         response_time = time.time() - start_time

#         # 적극성 점수 계산
#         proactivity_score = 0
#         if response_time < average_response_time:
#             proactivity_score += 1
#         else:
#             proactivity_score -= 1

#         if not emotion_change:
#             proactivity_score += 1
#         else:
#             proactivity_score -= 1

#         if proactivity_score >= 1:
#             proactivity = "높음"
#         else:
#             proactivity = "낮음"

#         dialogue_state.proactivity = proactivity

#         response = generate_response_second_phase(user_input, "gpt-3.5-turbo", dialogue_state)

#         # 응답 시간을 리스트에 추가
#         response_times.append(response_time)

#     dialogue_state.history[-1]['response'] = response

#     # 응답 데이터 생성
#     response_data = {
#         'timestamp': timestamp,
#         'response': response,
#         'dialogue_state': {
#             'intent': dialogue_state.intent,
#             'slots': dialogue_state.slots,
#             'emotion': dialogue_state.emotion,
#             'proactivity': dialogue_state.proactivity,
#             'consistency': dialogue_state.consistency,
#             'history': dialogue_state.history[-5:]  # 최근 5개 대화만 포함
#         }
#     }

#     return jsonify(response_data), 200

# def analyze_emotion(text, model_engine):
#     prompt = f"""
#     다음 문장의 감정을 분석해주세요.
#     문장: "{text}"
#     감정은 '긍정', '부정', '중립' 중 하나로 답변해주세요.
#     - 단순한 요청이나 정보 전달, 음식 주문과 관련된 문장은 '중립'으로 분류하세요.
#     - 망설임이나 고민을 나타내는 표현이 있어도, 부정적인 감정이 명확하지 않다면 '중립'으로 분류하세요.
#     """
#     try:
#         response = openai.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=10,
#             temperature=0,
#             n=1,
#             stop=None,
#         )
#         emotion = response.choices[0].message.content.strip()
#         if "긍정" in emotion:
#             return "긍정"
#         elif "부정" in emotion:
#             return "부정"
#         else:
#             return "중립"
#     except openai.error.OpenAIError as e:
#         print(f"Error during emotion analysis: {e}")
#         return "알 수 없음"

# def extract_intent_and_slots(user_input, model_engine):
#     prompt = f"""
#     다음 문장에서 사용자의 의도와 필요한 슬롯 정보를 추출해주세요.
#     문장: "{user_input}"
#     응답 형식:
#     Intent: [의도]
#     Slots:
#     [슬롯명1]: [값1]
#     [슬롯명2]: [값2]
#     만약 해당하지 않으면 '해당 없음'이라고 적어주세요.
#     """
#     try:
#         response = openai.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=150,
#             temperature=0,
#             n=1,
#             stop=None,
#         )
#         content = response.choices[0].message.content.strip()
#         # 응답 파싱
#         lines = content.split('\n')
#         intent = None
#         slots = {}
#         for line in lines:
#             if line.startswith('Intent:'):
#                 intent = line.replace('Intent:', '').strip()
#             elif ':' in line and not line.startswith('Intent:'):
#                 slot_name, slot_value = line.split(':', 1)
#                 slots[slot_name.strip()] = slot_value.strip()
#         return intent, slots
#     except openai.error.OpenAIError as e:
#         print(f"Error during intent and slot extraction: {e}")
#         return None, {}

# def most_common(lst):
#     if lst:
#         data = Counter(lst)
#         return data.most_common(1)[0][0]
#     else:
#         return None

# def check_consistency(dialogue_state):
#     # 최근 5개의 대화에서 의도, 슬롯, 감정의 빈도수 계산
#     intents = [turn['intent'] for turn in dialogue_state.history[:-1]]  # 현재 입력 제외
#     emotions = [turn['emotion'] for turn in dialogue_state.history[:-1]]
#     slots_list = [turn['slots'] for turn in dialogue_state.history[:-1]]

#     # 빈도수 계산
#     intent_counter = Counter(intents)
#     emotion_counter = Counter(emotions)

#     # 현재 입력
#     current_intent = dialogue_state.history[-1]['intent']
#     current_emotion = dialogue_state.history[-1]['emotion']
#     current_slots = dialogue_state.history[-1]['slots']

#     # 일치 여부 판단
#     match_count = 0
#     total_checks = 0

#     # 의도 비교
#     total_checks += 1
#     if current_intent in intent_counter:
#         match_count += 1

#     # 감정 비교
#     total_checks += 1
#     if current_emotion in emotion_counter:
#         match_count += 1

#     # 슬롯 비교 (슬롯 키와 값이 이전 대화에서 존재하는지 확인)
#     previous_slots = {}
#     for slots in slots_list:
#         for key, value in slots.items():
#             previous_slots.setdefault(key, set()).add(value)

#     total_checks += len(current_slots)
#     for key, value in current_slots.items():
#         if key in previous_slots and value in previous_slots[key]:
#             match_count += 1

#     # 일치율 계산
#     consistency_ratio = match_count / total_checks if total_checks > 0 else 0

#     # 임계값 설정 (예: 50% 이상 일치하면 일관성 있다고 판단)
#     threshold = 0.5
#     return consistency_ratio >= threshold

# def generate_response_first_phase(user_input, model_engine, dialogue_state):
#     prompt = f"""
#     당신은 친절하고 공감 능력이 뛰어난 AI 챗봇입니다.
#     사용자의 감정은 '{dialogue_state.emotion}'이며, 의도는 '{dialogue_state.intent}'입니다.
#     사용자의 말을 잘 이해하고, 자연스럽고 따뜻한 대화를 이어가세요.
#     응답은 두 문장으로 제한하세요.
#     사용자: "{user_input}"
#     """
#     try:
#         response = openai.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": prompt}
#             ],
#             max_tokens=100,
#             temperature=0.7,
#             n=1,
#             stop=None,
#         )
#         assistant_reply = response.choices[0].message.content.strip()
#         # 응답을 두 문장으로 제한
#         assistant_reply = '. '.join(assistant_reply.split('. ')[:2]).strip()
#         if not assistant_reply.endswith('.'):
#             assistant_reply += '.'
#         return assistant_reply
#     except openai.error.OpenAIError as e:
#         print(f"Error during response generation: {e}")
#         return "죄송하지만 응답을 생성할 수 없습니다."

# def generate_response_second_phase(user_input, model_engine, dialogue_state):
#     # 적극성과 일관성에 따른 프롬프트 생성
#     proactivity_flag = dialogue_state.proactivity == "높음"
#     consistency_flag = dialogue_state.consistency

#     if proactivity_flag and consistency_flag:
#         prompt = f"""
#         당신은 활기차고 공감 능력이 뛰어난 AI 챗봇입니다.
#         사용자의 감정은 '{dialogue_state.emotion}'이며, 의도는 '{dialogue_state.intent}'입니다.
#         대화의 흐름이 일관되므로 자연스럽게 대화를 이어가세요.
#         응답은 두 문장으로 제한하고, 의문형 문장을 사용하지 마세요.
#         사용자: "{user_input}"
#         """
#     elif proactivity_flag and not consistency_flag:
#         prompt = f"""
#         당신은 활기차고 공감 능력이 뛰어난 AI 챗봇입니다.
#         사용자의 감정은 '{dialogue_state.emotion}'이며, 의도는 '{dialogue_state.intent}'입니다.
#         대화의 흐름에 약간의 변화가 있으므로, 부드럽게 새로운 주제를 받아들이세요.
#         응답은 두 문장으로 제한하고, 의문형 문장을 사용하지 마세요.
#         사용자: "{user_input}"
#         """
#     elif not proactivity_flag and consistency_flag:
#         prompt = f"""
#         당신은 차분하고 사려 깊은 AI 챗봇입니다.
#         사용자의 감정은 '{dialogue_state.emotion}'이며, 의도는 '{dialogue_state.intent}'입니다.
#         대화의 흐름이 일관되므로 상세하고 정확한 답변을 제공하세요.
#         응답은 두 문장으로 제한하고, 의문형 문장을 사용하지 마세요.
#         사용자: "{user_input}"
#         """
#     else:
#         prompt = f"""
#         당신은 차분하고 사려 깊은 AI 챗봇입니다.
#         사용자의 감정은 '{dialogue_state.emotion}'이며, 의도는 '{dialogue_state.intent}'입니다.
#         대화의 흐름에 변화가 있으므로, 자연스럽게 새로운 주제로 대화를 이어가세요.
#         응답은 두 문장으로 제한하고, 의문형 문장을 사용하지 마세요.
#         사용자: "{user_input}"
#         """
#     try:
#         response = openai.chat.completions.create(
#             model=model_engine,
#             messages=[
#                 {"role": "system", "content": prompt}
#             ],
#             max_tokens=100,
#             temperature=0.7,
#             n=1,
#             stop=None,
#         )
#         assistant_reply = response.choices[0].message.content.strip()
#         # 응답을 두 문장으로 제한하고, 의문형 문장 제거
#         sentences = assistant_reply.split('. ')
#         filtered_sentences = [s for s in sentences if not s.strip().endswith('?')]
#         assistant_reply = '. '.join(filtered_sentences[:2]).strip()
#         if not assistant_reply.endswith('.'):
#             assistant_reply += '.'
#         return assistant_reply
#     except openai.error.OpenAIError as e:
#         print(f"Error during response generation: {e}")
#         return "죄송하지만 응답을 생성할 수 없습니다."

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

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
    def __init__(self, max_slots=5):
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
        대화의 흐름에 약간의 변화가 있으므로, 부드럽게 새로운 주제를 받아들이세요.
        사용자: "{user_input}"
        """
    elif not proactivity_flag and consistency_flag:
        prompt = f"""
        당신은 차분하고 사려 깊은 AI 챗봇입니다.
        사용자: "{user_input}"
        """
    else:
        prompt = f"""
        당신은 차분하고 사려 깊은 AI 챗봇입니다.
        대화의 흐름에 변화가 있으므로, 자연스럽게 새로운 주제로 대화를 이어가세요.
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)