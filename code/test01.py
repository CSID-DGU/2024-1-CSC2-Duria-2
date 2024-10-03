import os
import openai
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def main():
    # OpenAI API 키 설정
    openai.api_key = os.getenv('OPENAI_API_KEY')

    if not openai.api_key:
        print("Error: OpenAI API 키가 설정되지 않았습니다.")
        return

    # 모델 선택
    model_engine = "gpt-3.5-turbo"  # 또는 "gpt-4"

    print(f"{model_engine} 챗봇에 오신 것을 환영합니다! 종료하려면 'exit'를 입력하세요.\n")

    # 감정 분석 모델 로드 (beomi/KcELECTRA-base-v2022 사용)
    bert_model_name = "beomi/KcELECTRA-base-v2022"  # 한국어 감정 분석에 적합한 공개된 모델
    emotion_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name)
    emotion_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # 이전 대화 로그를 저장할 리스트
    responses_log = []

    while True:
        instruction = input("You: ")
        if instruction.lower() in ['exit', 'quit', '종료']:
            print("챗봇을 종료합니다. 이용해 주셔서 감사합니다.")
            break

        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 초기화
        previous_log = None  # previous_log 변수를 항상 초기화

        if responses_log:
            # 가장 유사한 이전 사용자의 대화를 선택하고 일관성 점수를 반환
            previous_log, similarity_score, time_weight = select_most_similar_conversation(instruction, responses_log)

            if previous_log:
                previous_user_input = previous_log['instruction']  # AI 응답이 아닌 사용자의 입력과 비교
                previous_emotion, current_emotion = check_emotion_and_negativity(previous_user_input, instruction,
                                                                                 emotion_model, emotion_tokenizer)

                # 감정 변화를 긍정, 부정, 중립으로 변환
                previous_emotion_state = get_emotion_label(previous_emotion)
                current_emotion_state = get_emotion_label(current_emotion)

                # 감정 변화가 있는지 체크
                emotion_change = abs(previous_emotion - current_emotion)

                # 일관성 점수 계산 (가중치 조정)
                consistency_score = (similarity_score * 0.8) + (time_weight * 0.15) - (emotion_change * 0.05)

                # 결과 출력 (현재 감정, 이전 감정 포함)
                print(f"현재 감정: {current_emotion_state} ({current_emotion}), 이전 감정: {previous_emotion_state} ({previous_emotion})")
                print(f"유사도: {similarity_score}, 타임스탬프 가중치: {time_weight}, 감정 변화: {emotion_change}")
                print(f"일관성 점수: {consistency_score:.2f}")

                # 일관성 판단 기준: 0.7 이상이면 일관성이 있다고 판단
                if consistency_score >= 0.7:
                    print("일관성이 있다고 판단되었습니다. 적극적인 응답을 생성합니다.")
                    consistency = 1  # 일관성이 있다고 판단
                else:
                    print("일관성이 없다고 판단되었습니다. 재질문을 통해 사실 관계를 파악합니다.")
                    consistency = 0  # 일관성이 없다고 판단
            else:
                consistency = 1  # 이전 대화가 없을 경우 일관성 분석을 하지 않고 진행
                print("이전에 대화한 기록이 없습니다.")
        else:
            # 이전 대화가 없을 경우 일관성 분석을 하지 않음
            consistency = 1
            print("이전에 대화한 기록이 없습니다. 일관성 분석을 생략하고 바로 응답을 생성합니다.")

        # 모델이 응답을 생성
        response = generate_response(instruction, model_engine, consistency)

        # 응답과 타임스탬프를 배열에 저장
        responses_log.append({'timestamp': timestamp, 'instruction': instruction, 'response': response})

        # 현재 대화와 가장 유사한 문장, 일관성 분석 결과를 출력
        print("\n[AI의 응답]")
        print(f"Timestamp: {timestamp}")
        print(f"질문: {instruction}")
        print(f"응답: {response}")
        if previous_log:
            print(f"가장 유사한 이전 대화: {previous_log['instruction']} (유사도: {similarity_score:.2f})")
            print(f"일관성 점수: {consistency_score:.2f}")
            print(f"이전 감정: {previous_emotion_state}, 현재 감정: {current_emotion_state}")
        print("\n" + "=" * 50 + "\n")


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
        try:
            outputs = emotion_model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1)

            # 감정 출력 디버깅용
            print(f"Logits: {logits}")
            print(f"Predicted emotion class: {predicted_class}")
        except Exception as e:
            print(f"Error: {e}")
            return 0, 0  # 오류가 발생하면 기본값으로 반환

    previous_emotion = predicted_class[0].item()
    current_emotion = predicted_class[1].item()

    return previous_emotion, current_emotion


def get_emotion_label(emotion_class):
    """ 감정 클래스를 긍정, 부정, 중립으로 변환하는 함수 """
    if emotion_class == 0:
        return "부정"
    elif emotion_class == 1:
        return "중립"
    elif emotion_class == 2:
        return "긍정"
    else:
        return "알 수 없음"


def generate_response(instruction, model_engine, consistency):
    # 일관성이 있을 때와 없을 때의 프롬프트 생성
    if consistency: # 일관성이 일치할 경우
        prompt = f'''당신은 친근하고 유머러스한 AI 챗봇입니다. 사용자의 질문에 대해 적극적이고 자신감 있게 답변해야 합니다. 
        만약 사용자의 질문과 맥락이 일치하거나 일관성이 있다고 판단되면, 확신을 가지고 답변하세요.

        사용자: {instruction}
        AI:'''
    else: # 일관성이 떨어질 경우
        prompt = f'''당신은 친절하고 사려 깊은 AI 챗봇입니다. 사용자의 질문에 대해 추가적인 정보를 요구하거나, 질문이 명확하지 않은 경우에는 재질문을 통해 확인을 하십시오.
        사용자의 질문과 이전 대화가 일치하지 않는다고 판단되면, "조금 더 구체적으로 설명해 주실 수 있나요?" 또는 "다른 의미로 말씀하신 것인가요?"와 같이 확인 질문을 하세요.

        사용자: {instruction}
        AI:'''

    try:
        # OpenAI API 호출
        response = openai.ChatCompletion.create(
            model=model_engine,
            messages=[
                {"role": "system", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            top_p=0.9,
            n=1,
            stop=None,
        )

        assistant_reply = response.choices[0].message.content.strip()
        return assistant_reply

    except openai.error.OpenAIError as e:
        print(f"Error: {e}")
        return "죄송하지만 응답을 생성할 수 없습니다."


if __name__ == "__main__":
    main()
