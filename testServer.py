from flask import Flask, request, jsonify
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import openai
import os

app = Flask(__name__)

# OpenAI API 설정
openai.api_key = os.getenv('OPENAI_API_KEY')

# 감정 분석 모델 로드 (beomi/KcELECTRA-base-v2022)
bert_model_name = "beomi/KcELECTRA-base-v2022"
emotion_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
emotion_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

# 대화 로그
responses_log = []

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    instruction = data.get('instruction', '')

    if not instruction:
        return jsonify({"error": "No input provided"}), 400

    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 일관성 및 감정 분석 초기화
    previous_log = None
    consistency = 1  # 기본적으로 일관성 있다고 설정

    if responses_log:
        # 이전 대화와 유사성을 분석
        previous_log, similarity_score, time_weight = select_most_similar_conversation(instruction, responses_log)
        if previous_log:
            previous_user_input = previous_log['instruction']
            previous_emotion, current_emotion = check_emotion_and_negativity(previous_user_input, instruction,
                                                                             emotion_model, emotion_tokenizer)
            emotion_change = abs(previous_emotion - current_emotion)
            consistency_score = (similarity_score * 0.8) + (time_weight * 0.15) - (emotion_change * 0.05)

            # 일관성 판단
            if consistency_score < 0.7:
                consistency = 0

    # 응답 생성
    response_text = generate_response(instruction, consistency)

    # 대화 로그 저장
    responses_log.append({'timestamp': timestamp, 'instruction': instruction, 'response': response_text})

    # 응답 반환
    return jsonify({
        'response': response_text,
        'similar_conversation': previous_log['instruction'] if previous_log else None,
        'consistency_score': consistency_score if previous_log else None,
        'emotion': current_emotion if previous_log else None
    }), 200, {'Content-Type': 'application/json; charset=utf-8'}


def select_most_similar_conversation(current_instruction, responses_log):
    if not responses_log:
        return None, 0, 0  # 유사한 대화가 없을 경우 일관성 0으로 반환

    # 타임스탬프 가중치 계산
    current_time = datetime.now()
    time_weights = []
    for log in responses_log:
        time_diff = (current_time - datetime.strptime(log['timestamp'], "%Y-%m-%d %H:%M:%S")).total_seconds()
        time_weights.append(max(1, 1 / (time_diff + 1)))

    # 코사인 유사도 계산
    vectorizer = TfidfVectorizer()
    all_user_inputs = [log['instruction'] for log in responses_log] + [current_instruction]
    tfidf_matrix = vectorizer.fit_transform(all_user_inputs)
    cosine_similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1]).flatten()

    # 유사도와 타임스탬프 가중치를 곱해 최종 가중치를 계산
    weighted_similarities = cosine_similarities * time_weights
    most_similar_index = weighted_similarities.argmax()

    # 가장 유사한 대화의 유사도 값 및 타임스탬프 가중치 반환
    return responses_log[most_similar_index], cosine_similarities[most_similar_index], time_weights[most_similar_index]

def check_emotion_and_negativity(previous_response, current_response, emotion_model, emotion_tokenizer):
    # 빈 입력 처리
    if not previous_response.strip() or not current_response.strip():
        return 0, 0  # 기본값으로 감정 일치(0, 0) 처리

    # 입력 텍스트 길이 제한 (예: 512 토큰)
    max_length = 512
    inputs = emotion_tokenizer(
        [previous_response[:max_length], current_response[:max_length]],
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=-1)

    previous_emotion = predicted_class[0].item()
    current_emotion = predicted_class[1].item()

    return previous_emotion, current_emotion

def generate_response(instruction, consistency):
    # 일관성이 있을 때와 없을 때의 프롬프트 생성
    if consistency:
        prompt = f'''당신은 친근하고 유머러스한 AI 챗봇입니다. 사용자의 질문에 대해 적극적이고 자신감 있게 답변해야 합니다. 
        만약 사용자의 질문과 맥락이 일치하거나 일관성이 있다고 판단되면, 확신을 가지고 답변하세요.

        사용자: {instruction}
        AI:'''
    else:
        prompt = f'''당신은 친절하고 사려 깊은 AI 챗봇입니다. 사용자의 질문에 대해 추가적인 정보를 요구하거나, 질문이 명확하지 않은 경우에는 재질문을 통해 확인을 하십시오.
        사용자의 질문과 이전 대화가 일치하지 않는다고 판단되면, "조금 더 구체적으로 설명해 주실 수 있나요?" 또는 "다른 의미로 말씀하신 것인가요?"와 같이 확인 질문을 하세요.

        사용자: {instruction}
        AI:'''

    try:
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # 사용할 모델 선택
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None,
        )

        assistant_reply = response.choices[0].message['content'].strip()
        return assistant_reply

    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)