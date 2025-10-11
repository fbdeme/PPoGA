import json


def calculate_total_time():
    """
    데이터셋 파일들을 읽어 총 질문 수를 계산하고,
    이를 바탕으로 전체 실험 소요 시간을 예측합니다.
    """

    files = {
        "CWQ": "/Users/jeonmingyu/Library/CloudStorage/GoogleDrive-jkmcoma7@gmail.com/My Drive/projects/CIKM/ppoga-project/data/cwq.json",
        "WebQSP": "/Users/jeonmingyu/Library/CloudStorage/GoogleDrive-jkmcoma7@gmail.com/My Drive/projects/CIKM/ppoga-project/data/WebQSP.json",
        "GrailQA": "/Users/jeonmingyu/Library/CloudStorage/GoogleDrive-jkmcoma7@gmail.com/My Drive/projects/CIKM/ppoga-project/data/grailqa.json",
    }

    total_questions = 0

    print("데이터셋별 질문 수:")
    try:
        for name, path in files.items():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                count = len(data)
                print(f"- {name}: {count}개")
                total_questions += count

        print("-" * 30)
        print(f"총 질문 수: {total_questions}개")

        # 시간 계산
        avg_time_per_question = 50  # 초
        total_seconds = total_questions * avg_time_per_question

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        print(
            f"\n총 예상 소요 시간 (질문당 50초 기준): {hours}시간 {minutes}분 {seconds}초"
        )

    except FileNotFoundError as e:
        print(f"\n[오류] 파일을 찾을 수 없습니다: {e}")
        print("이 스크립트가 데이터 파일들과 같은 폴더에 있는지 확인해주세요.")
    except Exception as e:
        print(f"\n[오류] 예상치 못한 문제가 발생했습니다: {e}")


if __name__ == "__main__":
    calculate_total_time()
