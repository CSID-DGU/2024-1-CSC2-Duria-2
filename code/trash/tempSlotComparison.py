import requests
import json
import csv

# 서버 주소 설정
SERVER_URL = 'http://localhost:5000/chat'

# CSV 파일 경로
CSV_FILE = 'test_cases.csv'

def load_test_cases_from_csv(file_path):
    """
    CSV 파일에서 테스트 케이스를 로드합니다.
    """
    test_cases = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            user_input = row['user_input']
            expected_consistency = row['expected_consistency']
            # expected_consistency를 적절한 타입으로 변환
            if expected_consistency == 'True':
                expected_consistency = True
            elif expected_consistency == 'False':
                expected_consistency = False
            elif expected_consistency == 'None':
                expected_consistency = None
            else:
                # 예상 일관성이 지정되지 않은 경우 None으로 설정
                expected_consistency = None
            test_cases.append({
                'user_input': user_input,
                'expected_consistency': expected_consistency
            })
    return test_cases

def send_message(user_input):
    """
    서버에 메시지를 보내고 응답을 반환합니다.
    """
    payload = {
        'user_input': user_input
    }
    headers = {
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(SERVER_URL, data=json.dumps(payload), headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: 서버 응답 상태 코드 {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def run_tests():
    """
    테스트 케이스를 실행하고 결과를 출력합니다.
    """
    print("테스트 시작...\n")

    # 테스트 케이스 로드
    test_cases = load_test_cases_from_csv(CSV_FILE)

    for idx, test in enumerate(test_cases, 1):
        user_input = test['user_input']
        expected = test['expected_consistency']
        response = send_message(user_input)
        
        if response is None:
            print(f"테스트 {idx}: 서버 응답 없음")
            continue

        # 챗봇 응답 추출
        bot_response = response.get('response', '')
    
        # 일관성 값 추출
        dialogue_state = response.get('dialogue_state', {})
        consistency = dialogue_state.get('consistency', None)

        # 예상 값과 비교
        if expected is None:
            # 초기 대화의 경우, consistency는 판단되지 않음
            result = "N/A"
            match = True  # 일치 여부를 판단하지 않음
        else:
            result = consistency
            match = (consistency == expected)

        # 챗봇의 원문 응답(JSON 데이터)를 보기 좋게 포매팅
        raw_response = json.dumps(response, ensure_ascii=False, indent=2)

        # 결과 출력
        status = "PASS" if match else "FAIL"
        print(f"테스트 {idx}:")
        print(f"  사용자 입력: {user_input}")
        print(f"  챗봇 응답: {bot_response}")
        print(f"  예상 일관성: {expected}")
        print(f"  서버 일관성: {consistency}")
        print(f"  결과: {status}")
        print(f"  챗봇 원문 응답(JSON):\n{raw_response}\n")  # 챗봇의 원문 응답 출력

if __name__ == '__main__':
    run_tests()
