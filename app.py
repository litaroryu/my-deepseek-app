import streamlit as st
import requests
import json
from typing import Dict, List, Generator
import re
import time

# 페이지 설정
st.set_page_config(
    page_title="DeepSeek-R1 Chatbot",
    page_icon="🧠",
    layout="wide"
)

# Ollama API 설정
OLLAMA_BASE_URL = "http://localhost:11434"
DEEPSEEK_MODEL = "deepseek-r1:8b"

class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url

    def check_connection(self) -> bool:
        """Ollama 서버 연결 확인"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_models(self) -> List[str]:
        """사용 가능한 모델 목록 가져오기"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            return []
        except requests.exceptions.RequestException:
            return []

    def chat_stream(self, model: str, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        """스트리밍 채팅 응답"""
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120  # DeepSeek-R1은 추론 시간이 길 수 있음
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if 'message' in data and 'content' in data['message']:
                                yield data['message']['content']
                            if data.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"오류 발생: HTTP {response.status_code}"

        except requests.exceptions.RequestException as e:
            yield f"연결 오류: {str(e)}"

def parse_deepseek_response(response: str) -> tuple[str, str]:
    """DeepSeek-R1 응답에서 추론 과정과 최종 답변 분리"""
    # <think> 태그로 감싸진 추론 과정 추출
    thinking_pattern = r'<think>(.*?)</think>'
    thinking_match = re.search(thinking_pattern, response, re.DOTALL)

    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # <think> 태그 제거한 나머지가 최종 답변
        final_answer = re.sub(thinking_pattern, '', response, flags=re.DOTALL).strip()
    else:
        # <think> 태그가 없는 경우 전체를 최종 답변으로 처리
        thinking = ""
        final_answer = response.strip()

    return thinking, final_answer

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = OllamaClient()

if "show_thinking" not in st.session_state:
    st.session_state.show_thinking = True

def main():
    st.title("🧠 DeepSeek-R1 Chatbot")
    st.markdown("**추론 과정을 보여주는 AI 모델과 채팅하기**")

    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        # 연결 상태 확인
        if st.button("🔗 연결 확인"):
            with st.spinner("Ollama 서버 연결 확인 중..."):
                is_connected = st.session_state.ollama_client.check_connection()
                if is_connected:
                    st.success("✅ Ollama 서버 연결됨")

                    # DeepSeek-R1 모델 확인
                    models = st.session_state.ollama_client.get_models()
                    if DEEPSEEK_MODEL in models:
                        st.success(f"✅ {DEEPSEEK_MODEL} 모델 사용 가능")
                    else:
                        st.warning(f"⚠️ {DEEPSEEK_MODEL} 모델이 없습니다")
                        st.code(f"ollama pull {DEEPSEEK_MODEL}")
                else:
                    st.error("❌ Ollama 서버에 연결할 수 없습니다")

        st.divider()

        # DeepSeek-R1 특화 설정
        st.subheader("🧠 DeepSeek-R1 옵션")
        show_thinking = st.checkbox("🤔 추론 과정 보기", value=st.session_state.show_thinking)
        st.session_state.show_thinking = show_thinking

        st.info("💡 DeepSeek-R1은 답변 전에 추론 과정을 거칩니다. 시간이 조금 걸릴 수 있어요!")

        st.divider()

        # 모델 파라미터
        st.subheader("🎛️ 생성 옵션")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
        max_tokens = st.slider("Max Tokens", 500, 8000, 4000, 500)
        top_p = st.slider("Top P", 0.1, 1.0, 0.8, 0.1)

        st.divider()

        # 프롬프트 템플릿
        st.subheader("📝 질문 템플릿")
        template_options = {
            "일반 대화": "",
            "수학 문제": "다음 수학 문제를 단계별로 풀어주세요:\n",
            "코딩 문제": "다음 프로그래밍 문제를 해결해주세요:\n",
            "논리적 추론": "다음 문제를 논리적으로 분석해주세요:\n",
            "창의적 글쓰기": "다음 주제로 창의적인 글을 써주세요:\n"
        }

        selected_template = st.selectbox("질문 유형 선택:", list(template_options.keys()))

        st.divider()

        # 대화 기록 관리
        if st.button("🗑️ 대화 기록 삭제"):
            st.session_state.messages = []
            st.rerun()

        # 대화 통계
        if st.session_state.messages:
            st.subheader("📊 대화 통계")
            user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            st.metric("사용자 메시지", user_msgs)
            st.metric("AI 응답", assistant_msgs)

    # 메인 채팅 영역
    chat_container = st.container()

    with chat_container:
        # 대화 기록 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "thinking" in message:
                    # 추론 과정이 있는 경우
                    if st.session_state.show_thinking and message["thinking"]:
                        with st.expander("🤔 추론 과정 보기", expanded=False):
                            st.markdown(f"```\n{message['thinking']}\n```")
                    st.markdown(message["content"])
                else:
                    st.markdown(message["content"])

        # 사용자 입력
        prompt_input = st.chat_input("메시지를 입력하세요...")

        if prompt_input:
            # 템플릿 적용
            template_prefix = template_options[selected_template]
            full_prompt = template_prefix + prompt_input if template_prefix else prompt_input

            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": full_prompt})

            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.markdown(full_prompt)

            # AI 응답 생성
            with st.chat_message("assistant"):
                # 진행 상황 표시
                progress_placeholder = st.empty()
                progress_placeholder.info("🧠 DeepSeek-R1이 생각하고 있습니다...")

                message_placeholder = st.empty()
                thinking_placeholder = st.empty()
                full_response = ""

                # 스트리밍 응답
                try:
                    start_time = time.time()
                    response_stream = st.session_state.ollama_client.chat_stream(
                        model=DEEPSEEK_MODEL,
                        messages=st.session_state.messages,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens,
                            "top_p": top_p,
                        }
                    )

                    first_chunk_received = False
                    for chunk in response_stream:
                        if not first_chunk_received:
                            progress_placeholder.empty()  # 진행 상황 메시지 제거
                            first_chunk_received = True

                        full_response += chunk

                        # 실시간으로 추론 과정과 최종 답변 분리
                        thinking, final_answer = parse_deepseek_response(full_response)

                        # 추론 과정 표시 (옵션에 따라)
                        if st.session_state.show_thinking and thinking:
                            with thinking_placeholder.expander("🤔 추론 과정", expanded=True):
                                st.markdown(f"```\n{thinking}\n```")

                        # 최종 답변 표시
                        if final_answer:
                            message_placeholder.markdown(final_answer + "▌")

                    # 최종 정리
                    thinking, final_answer = parse_deepseek_response(full_response)

                    if st.session_state.show_thinking and thinking:
                        with thinking_placeholder.expander("🤔 추론 과정", expanded=False):
                            st.markdown(f"```\n{thinking}\n```")

                    message_placeholder.markdown(final_answer if final_answer else full_response)

                    # 응답 시간 표시
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.caption(f"⏱️ 응답 시간: {response_time:.1f}초")

                except Exception as e:
                    progress_placeholder.empty()
                    error_msg = f"오류가 발생했습니다: {str(e)}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
                    thinking = ""
                    final_answer = error_msg

            # 응답을 대화 기록에 추가 (추론 과정 포함)
            thinking, final_answer = parse_deepseek_response(full_response) if not full_response.startswith("오류") else ("", full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer if final_answer else full_response,
                "thinking": thinking
            })

    # 샘플 질문 제안
    st.subheader("💡 샘플 질문")
    col1, col2, col3 = st.columns(3)

    sample_questions = [
        "25 × 37을 단계별로 계산해주세요",
        "Python으로 피보나치 수열을 구현하는 방법",
        "왜 하늘은 파란색일까요?",
        "창의적인 단편소설 아이디어를 제안해주세요",
        "기후변화의 주요 원인 3가지는?",
        "간단한 암호화 알고리즘을 설명해주세요"
    ]

    for i, question in enumerate(sample_questions):
        col = [col1, col2, col3][i % 3]
        if col.button(f"❓ {question[:20]}...", key=f"sample_{i}"):
            st.session_state.sample_question = question
            st.rerun()

    # 샘플 질문 자동 입력
    if hasattr(st.session_state, 'sample_question'):
        st.session_state.messages.append({"role": "user", "content": st.session_state.sample_question})
        del st.session_state.sample_question
        st.rerun()

    # 하단 정보
    with st.expander("ℹ️ DeepSeek-R1 사용 가이드"):
        st.markdown("""
        ### 🧠 DeepSeek-R1 모델 특징:
        - **추론 기반**: 답변하기 전에 단계별 사고 과정을 거칩니다
        - **투명성**: 어떻게 결론에 도달했는지 보여줍니다
        - **정확성**: 복잡한 문제도 체계적으로 해결합니다

        ### 📥 설치 방법:
        ```bash
        # DeepSeek-R1 모델 다운로드 (약 4.7GB)
        ollama pull deepseek-r1:8b

        # Ollama 서버 실행
        ollama serve
        ```

        ### 💡 활용 팁:
        - **수학 문제**: 단계별 계산 과정을 명확히 보여줍니다
        - **코딩 문제**: 알고리즘 설계부터 구현까지 설명합니다
        - **논리 추론**: 복잡한 문제를 체계적으로 분석합니다
        - **창의적 작업**: 아이디어 발전 과정을 투명하게 공개합니다

        ### ⚙️ 최적 설정:
        - Temperature: 0.1-0.5 (논리적 답변) / 0.6-0.8 (창의적 답변)
        - Max Tokens: 4000+ (충분한 추론 공간 제공)
        """)

if __name__ == "__main__":
    main()