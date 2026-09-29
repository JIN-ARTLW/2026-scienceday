"""결과 뷰어 실행.

uv run python scripts/viewer.py            # 창 열기
uv run python scripts/viewer.py --export   # 창 없이 최신 결과 그림을 PNG로 저장
"""

from orbital_decay.viewer import main

if __name__ == "__main__":
    main()
