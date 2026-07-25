from dotenv import load_dotenv
import os
import logging
import sys

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables from .env file (for local dev)
# On Vercel, environment variables are already set in the environment
load_dotenv()

# Get API key from environment (Vercel sets this automatically)
openai_api_key = os.getenv("OPENAI_API_KEY")

# Lazy import OpenAI to avoid import errors at module load time
_client = None

def _get_client():
    """Get or create OpenAI client lazily."""
    global _client
    if _client is not None:
        return _client
    if openai_api_key:
        from openai import OpenAI
        _client = OpenAI(api_key=openai_api_key)
    return _client

# Check if API key exists - log warning but don't crash
if not openai_api_key:
    logger.warning("OPENAI_API_KEY environment variable not set - AI features will be disabled")

# Promotional/spam content patterns to filter
PROMOTIONAL_PATTERNS = [
    r"subscribe",
    r"đăng\s*ký\s*(kênh|ngay|channel|subscribe)",
    r"kênh\s+\w+",
    r"theo\s+dõi",
    r"follow\s+(me|channel)",
    r"like\s+and\s+subscribe",
    r"nhấn\s+(like|đăng\s*ký)",
    r"clip\s+(hay|interesting)",
    r"xem\s+(hết|full)",
    r"link\s+(bio|description)",
    r"donate",
    r"Ủng hộ",
    r"mua\s+xe",
    r"quảng\s*cáo",
    r"khuyến\s*mãi",
    r"giảm\s*giá",
    r"trúng\s*thưởng",
    r"click\s*here",
    r"click\s*link",
]

# Compile regex patterns for efficiency
import re
import time
import threading

_promotional_regex = re.compile(
    "|".join(PROMOTIONAL_PATTERNS), re.IGNORECASE | re.VERBOSE
)

_reply_cache = {}
_cache_ttl = 300

_web_cache = {}
_web_cache_ttl = 86400
_web_prefetch_done = False
_web_prefetch_lock = threading.Lock()


def _is_promotional_content(text: str) -> bool:
    """
    Check if the text appears to be promotional/spam content
    rather than a genuine user query.
    """
    if not text or len(text.strip()) < 5:
        return True

    if _promotional_regex.search(text):
        logger.info(f"Filtered promotional content: {text[:50]}...")
        return True

    return False


def _prefetch_websites():
    """Pre-fetch and cache website content in background on startup."""
    global _web_cache, _web_prefetch_done

    with _web_prefetch_lock:
        if _web_prefetch_done:
            return
        _web_prefetch_done = True

    urls = [
        "https://phuongtanhung.gov.vn",
        "https://phuongtanhung.org",
        "https://dichvucong.gov.vn/p/home/dvc-dich-vu-cong-truc-tuyen-ds.html?pCoQuanId=411312#mainTitle",
    ]

    def fetch_in_background():
        for url in urls:
            try:
                content = _fetch_website_content(url, max_chars=1000)
                if content:
                    _web_cache[url] = content
                    logger.info(f"Cached website content: {url[:40]}...")
            except Exception:
                pass

    thread = threading.Thread(target=fetch_in_background, daemon=True)
    thread.start()
    logger.info("Started background website prefetch")


def _get_cached_web_context(text):
    """Get website context only for relevant questions using cache or fast fetch."""
    if not _is_website_question(text) and not _is_admin_service_question(text):
        return ""

    web_context = ""
    urls = [
        "https://phuongtanhung.gov.vn",
        "https://phuongtanhung.org",
        "https://dichvucong.gov.vn/p/home/dvc-dich-vu-cong-truc-tuyen-ds.html?pCoQuanId=411312#mainTitle",
    ]

    for url in urls:
        if url in _web_cache:
            web_context += f"\n\n[Nguồn: {url}]\n{_web_cache[url]}"
        elif _is_website_question(text) or _is_admin_service_question(text):
            content = _fetch_website_content(url, max_chars=1000)
            if content:
                _web_cache[url] = content
                web_context += f"\n\n[Nguồn: {url}]\n{content}"

    if web_context:
        logger.info(f"Added website content for: {text[:30]}...")

    return web_context


# Initialize knowledge retriever
_knowledge_loaded = False


def _load_knowledge():
    """Load knowledge base on first use."""
    global _knowledge_loaded
    if _knowledge_loaded:
        return

    try:
        # Import with correct relative path - try both for local vs deployment
        try:
            from backend.knowledge.retriever import retriever
        except ModuleNotFoundError:
            from knowledge.retriever import retriever

        success = retriever.load_knowledge()
        if success:
            logger.info(
                f"Knowledge base loaded with {len(retriever.documents)} documents"
            )
            _knowledge_loaded = True
        else:
            # Loading failed, don't mark as loaded to allow retry
            logger.warning("Knowledge base load returned False, will retry")
    except Exception as e:
        logger.warning(f"Failed to load knowledge base: {e}")
        import traceback

        traceback.print_exc()
        # Don't set _knowledge_loaded to True so it can retry


# Keywords that indicate user is asking about the AI system itself
AI_QUESTION_KEYWORDS = [
    "ai là gì",
    "bạn là ai",
    "em là ai",
    "who are you",
    "what are you",
    "who made",
    "ai made",
    "who created",
    "created by",
    "made by",
    "được ai làm",
    "do ai làm",
    "ai tạo",
    "ai xây dựng",
    "ai phát triển",
    "hệ thống ai",
    "về ai",
    "thông tin về",
    "giới thiệu",
    "giới thiệu về",
    "người tạo",
    "người làm",
    "lập trình",
    " Developer",
    "developer",
    "programmer",
    "programmed",
]

LOGIN_GUIDE_KEYWORDS = [
    "hướng dẫn đăng nhập",
    "sổ tay đảng viên điện tử",
    "đăng nhập sổ tay đảng viên",
    "sổ tay điện tử",
    "đảng viên điện tử",
    "đăng nhập sổ tay",
]

LOGIN_GUIDE_URL = "https://drive.google.com/file/d/1AHISbljI4IS_nwqjpXaB_O7cKarAK3oH/view?usp=drive_link"


def _is_login_guide_question(text: str) -> bool:
    text_lower = text.lower()
    for keyword in LOGIN_GUIDE_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def _load_ai_system_info():
    """Load the AI system info file content."""
    try:
        # Try multiple paths for the AI system info file
        possible_paths = [
            os.path.join(
                project_root, "backend", "knowledge", "data", "HỆ THỐNG AI.txt"
            ),
            os.path.join(project_root, "knowledge", "data", "HỆ THỐNG AI.txt"),
            os.path.join(project_root, "knowledge", "HỆ THỐNG AI.txt"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    logger.info(f"Loaded AI system info from: {path}")
                    return content

        logger.warning("AI system info file not found")
        return None
    except Exception as e:
        logger.warning(f"Failed to load AI system info: {e}")
        return None


def _is_ai_question(text):
    """Check if the question is about the AI system itself."""
    text_lower = text.lower()
    for keyword in AI_QUESTION_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def get_reply(text):
    """
    Get AI reply using knowledge base + website search.
    Always searches both knowledge and the two websites.
    """
    openai_client = _get_client()
    if openai_client is None:
        logger.warning("OpenAI client not configured - returning fallback response")
        return "Xin lỗi, hệ thống AI chưa được cấu hình. Vui lòng liên hệ quản trị viên."

    if _is_promotional_content(text):
        return "Xin lỗi, tôi không nghe rõ. Bạn có thể nói lại không?"

    if _is_login_guide_question(text):
        logger.info(f"Detected login guide question: {text[:50]}...")
        return (
            "Vui lòng xem video hướng dẫn đăng nhập sổ tay đảng viên điện tử tại đây: "
            + LOGIN_GUIDE_URL
        )

    cache_key = text.strip().lower()
    if cache_key in _reply_cache:
        cached = _reply_cache[cache_key]
        if time.time() - cached["time"] < _cache_ttl:
            logger.info(f"Returning cached reply for: {text[:30]}...")
            return cached["reply"]

    if _is_ai_question(text):
        logger.info(f"Detected AI question: {text[:50]}...")
        ai_info = _load_ai_system_info()

        if ai_info:
            try:
                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là một trợ lý ảo anime dễ thương tên là Đoàn Viên. "
                            "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu. "
                            "Sử dụng THÔNG TIN được cung cấp để trả lời câu hỏi về hệ thống AI.",
                        },
                        {"role": "system", "content": "THÔNG TIN:\n" + ai_info},
                        {"role": "user", "content": text},
                    ],
                )
                reply = response.choices[0].message.content
                _reply_cache[cache_key] = {"reply": reply, "time": time.time()}
                return reply
            except Exception as e:
                logger.warning(f"Failed to get AI info response: {e}")

    _load_knowledge()

    try:
        try:
            from backend.knowledge.retriever import retriever
        except ModuleNotFoundError:
            from knowledge.retriever import retriever

        logger.info(f"Knowledge base loaded: {retriever.has_knowledge()}")
        logger.info(f"Number of documents: {len(retriever.documents)}")

        if retriever.has_knowledge():
            context, found = retriever.get_answer_context(text, max_chars=5000)

            logger.info(
                f"Search result for '{text[:30]}...': found={found}, context_len={len(context)}"
            )

            if context:
                logger.info(f"Found relevant knowledge for: {text[:50]}...")
                logger.info(f"Context preview: {context[:200]}...")

                combined_context = context
                web_context = _get_cached_web_context(text)
                if web_context:
                    combined_context += web_context

                response = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "Bạn là một trợ lý ảo anime dễ thương tên là Đoàn Viên. "
                            "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu. "
                            "Sử dụng THÔNG TIN TÀI LIỆU được cung cấp để trả lời câu hỏi. "
                            "Nếu thông tin không đủ, hãy nói rằng bạn không có thông tin đó và gợi ý liên hệ cơ quan chức năng. "
                            "Khi trả lời về một người cụ thể, hãy dùng đầy đủ họ tên của người đó như trong tài liệu. "
                            "Nếu có nhiều người có họ tên giống hoặc gần giống nhau (ví dụ: cùng họ và tên đệm), hãy kiểm tra tên riêng cuối cùng có khớp chính xác với câu hỏi hay tài liệu không. "
                            "Nếu không chắc chắn người nào là đúng, hãy nói rõ ràng và liệt kê các trường hợp có thể.",
                        },
                        {
                            "role": "system",
                            "content": f"THÔNG TIN TÀI LIỆU:\n{combined_context}",
                        },
                        {"role": "user", "content": text},
                    ],
                )

                reply = response.choices[0].message.content
                _reply_cache[cache_key] = {"reply": reply, "time": time.time()}
                return reply
            else:
                logger.info(
                    f"No relevant context found in knowledge base for: {text[:30]}..."
                )
    except Exception as e:
        logger.warning(f"Knowledge search failed: {e}")
        import traceback

        traceback.print_exc()

    google_results = _search_google(text)

    if google_results:
        logger.info(f"Google search found results, generating answer...")
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Bạn là một trợ lý ảo anime dễ thương tên là Đoàn Viên. "
                        "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu. "
                        "Dựa vào THÔNG TIN TÌM THẤY từ Google dưới đây để trả lời câu hỏi. "
                        "Nếu tìm thấy thông tin liên quan, hãy tóm tắt lại cho người dùng. "
                        "Nếu không có thông tin trực tiếp, hãy chia sẻ các kết quả tìm kiếm "
                        "và gợi ý người dùng xem thêm tại các nguồn này.",
                    },
                    {
                        "role": "system",
                        "content": f"THÔNG TIN TÌM THẤY TỪ GOOGLE:\n{google_results}",
                    },
                    {"role": "user", "content": text},
                ],
            )
            reply = response.choices[0].message.content
            _reply_cache[cache_key] = {"reply": reply, "time": time.time()}
            return reply
        except Exception as e:
            logger.warning(f"Failed to generate answer from Google results: {e}")

    logger.info(f"Using OpenAI fallback for: {text[:50]}...")

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Bạn là một trợ lý ảo anime dễ thương. "
                "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu.",
            },
            {"role": "user", "content": text},
        ],
    )

    reply = response.choices[0].message.content
    _reply_cache[cache_key] = {"reply": reply, "time": time.time()}
    return reply


# Website keywords that trigger web search
WEBSITE_KEYWORDS = [
    "phuongtanhung.gov.vn",
    "phuongtanhung.org",
    "website phường",
    "trang web phường",
    "phuongtanhung",
    "gov.vn",
    "org.vn",
    "dichvucong.gov.vn",
    "dịch vụ công",
    "thủ tục hành chính",
]

# Legal document keywords - trigger thu vienphapluat.vn search
LEGAL_DOCUMENT_KEYWORDS = [
    "quyết định",
    "qđ",
    "thông tư",
    "nghị định",
    "ngân sách",
    "hợp đồng",
    "luật",
    "bộ luật",
    "nghị quyết",
    "quy chế",
    "quy định",
    "thông báo",
    "công văn",
    "văn bản",
    "pháp lệnh",
    "lệnh",
]

# Administrative service keywords
ADMIN_SERVICE_KEYWORDS = [
    "dịch vụ công",
    "thủ tục hành chính",
    "đăng ký",
    "cấp giấy",
    "làm giấy",
    "xin giấy",
    "hộ khẩu",
    "tạm trú",
    "thường trú",
    "chứng minh",
    "căn cước",
    "hộ chiếu",
    "đăng ký kinh doanh",
    "thành lập",
    "giải thể",
    "kết hôn",
    "ly hôn",
    "khai sinh",
    "khai tử",
    "nhận con",
    "nhận nuôi",
    "chuyển户口",
]


def _is_website_question(text):
    """Check if the question is about the websites."""
    text_lower = text.lower()
    for keyword in WEBSITE_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def _fetch_website_content(url, max_chars=5000):
    """Fetch content from a website with better headers."""
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "vi,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove scripts, styles, nav, footer, header
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()

            # Get title
            title = ""
            if soup.title:
                title = soup.title.string or ""

            # Get main content
            main_content = ""

            # Try common content containers
            for selector in [
                "main",
                "article",
                ".content",
                "#content",
                ".post-content",
                ".article-content",
            ]:
                elements = soup.select(selector)
                if elements:
                    for elem in elements:
                        text = elem.get_text(separator=" ", strip=True)
                        if len(text) > 100:
                            main_content += text + " "

            # If no selector worked, get body text
            if not main_content:
                body = soup.find("body")
                if body:
                    main_content = body.get_text(separator=" ", strip=True)

            # Combine title and content
            full_text = f"{title} {main_content}" if title else main_content

            # Clean up extra whitespace
            import re

            full_text = re.sub(r"\s+", " ", full_text).strip()

            return full_text[:max_chars] if full_text else ""

        elif response.status_code == 403:
            logger.warning(f"Access forbidden 403 for {url} - site blocks automated requests")
            return f"[Không thể truy cập: {url} - Website từ chối truy cập tự động]"

        else:
            logger.warning(f"Failed to fetch {url}: status {response.status_code}")
            return f"[Lỗi {response.status_code}: {url}]"


    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return f"[Lỗi kết nối: {url}]"

def _is_admin_service_question(text):
    """Check if the question is about administrative services."""
    text_lower = text.lower()
    for keyword in ADMIN_SERVICE_KEYWORDS:
        if keyword.lower() in text_lower:
            return True
    return False


def _is_legal_document_question(text):
    """Check if the question is about legal documents (decrees, decisions, laws, etc.)."""
    text_lower = text.lower()
    for keyword in LEGAL_DOCUMENT_KEYWORDS:
        if keyword in text_lower:
            return True
    return False


def _search_dichvucong_service(query, max_chars=3000):
    """
    Search for administrative service on dichvucong.gov.vn.
    Uses the website's search functionality to find relevant services.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        import urllib.parse

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "vi,en-US;q=0.7,en;q=0.3",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Referer": "https://dichvucong.gov.vn/",
        }

        base_url = (
            "https://dichvucong.gov.vn/p/home/dvc-dich-vu-cong-truc-tuyen-ds.html"
        )
        search_params = f"pCoQuanId=411312&keyword={urllib.parse.quote(query)}"

        search_url = f"{base_url}?{search_params}"
        response = requests.get(
            search_url, headers=headers, timeout=5, allow_redirects=True
        )

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")

            results_text = ""

            service_items = soup.select(
                ".service-item, .dichvu-item, .list-item, tr[itemscope], .tinydichvu"
            )
            if service_items:
                for item in service_items[:5]:
                    text = item.get_text(separator=" ", strip=True)
                    if len(text) > 20:
                        results_text += text + "\n"
            else:
                main_content = soup.select(
                    "main, article, .content, #content, .main-content"
                )
                for elem in main_content:
                    text = elem.get_text(separator=" ", strip=True)
                    if len(text) > 50:
                        results_text += text[:max_chars]
                        break

            if not results_text:
                results_text = soup.get_text(separator=" ", strip=True)
                results_text = results_text[:max_chars] if results_text else ""

            import re

            results_text = re.sub(r"\s+", " ", results_text).strip()

            if results_text:
                logger.info(
                    f"Found service info for query '{query}': {len(results_text)} chars"
                )
                return results_text

        logger.warning(
            f"Failed to search dichvucong.gov.vn: status {response.status_code}"
        )
        return ""

    except Exception as e:
        logger.warning(f"Failed to search dichvucong.gov.vn: {e}")
        return ""


def _get_admin_service_answer(text):
    """Get answer about administrative services from dichvucong.gov.vn."""
    query = text

    service_content = _search_dichvucong_service(query, max_chars=4000)

    if not service_content:
        return None

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một trợ lý ảo anime dễ thương tên là Đoàn Viên. "
                    "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu. "
                    "Sử dụng THÔNG TIN TỪ TRANG WEB dịch vụ công được cung cấp để trả lời câu hỏi. "
                    "Nếu thông tin không đủ, hãy nói rằng bạn không tìm thấy thông tin đó và gợi ý liên hệ cơ quan chức năng.",
                },
                {
                    "role": "system",
                    "content": f"THÔNG TIN TỪ TRANG DỊCH VỤ CÔNG:\n{service_content}",
                },
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Failed to get admin service response: {e}")
        return None


def _get_website_answer(text):
    """Get answer about the websites by fetching and analyzing content."""
    text_lower = text.lower()

    if _is_admin_service_question(text):
        return _get_admin_service_answer(text)

    urls_to_check = []
    if "phuongtanhung.gov.vn" in text_lower or "phuongtanhung.org" in text_lower:
        if "gov.vn" in text_lower:
            urls_to_check.append("https://phuongtanhung.gov.vn")
        if "org" in text_lower:
            urls_to_check.append("https://phuongtanhung.org")

    for url in [
        "https://dichvucong.gov.vn/p/home/dvc-dich-vu-cong-truc-tuyen-ds.html?pCoQuanId=411312#mainTitle",
    ]:
        if (
            "dichvucong.gov.vn" in text_lower
            or "dịch vụ công" in text_lower
            or _is_admin_service_question(text)
        ):
            urls_to_check.append(url)

    if not urls_to_check:
        urls_to_check = ["https://phuongtanhung.gov.vn", "https://phuongtanhung.org"]

    all_content = []
    for url in urls_to_check:
        content = _fetch_website_content(url)
        if content:
            all_content.append(f"[Nguồn: {url}]\n{content}")

    if not all_content:
        return None

    context = "\n\n".join(all_content)

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là một trợ lý ảo anime dễ thương tên là Đoàn Viên. "
                    "Hãy trả lời bằng tiếng Việt tự nhiên, ngắn gọn, dễ hiểu. "
                    "Sử dụng THÔNG TIN TỪ TRANG WEB được cung cấp để trả lời câu hỏi. "
                    "Nếu thông tin không đủ, hãy nói rằng bạn không tìm thấy thông tin đó.",
                },
                {"role": "system", "content": f"THÔNG TIN TỪ TRANG WEB:\n{context}"},
                {"role": "user", "content": text},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.warning(f"Failed to get website response: {e}")
        return None


def _search_google(query, max_results=8):
    """Search Google for information as last resort."""
    try:
        import requests
        from bs4 import BeautifulSoup
        import urllib.parse
        import re
        
        # Prepare search query
        search_query = urllib.parse.quote(query)
        url = f"https://www.google.com/search?q={search_query}&num={max_results}&hl=vi&gl=VN"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        response = requests.get(url, headers=headers, timeout=5, allow_redirects=True)
        
        if response.status_code != 200:
            logger.warning(f"Google search failed with status: {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract search results
        results = []
        
        # Find all search result divs
        for g in soup.find_all("div", class_=["g", "tF2Cxc"]):
            try:
                # Try to find title and link
                link_element = g.find("a")
                if not link_element:
                    continue
                
                link = link_element.get("href", "")
                if not link.startswith("http"):
                    continue
                
                # Get title
                title_element = g.find("h3")
                title = title_element.get_text() if title_element else "Không có tiêu đề"
                
                # Get snippet
                snippet_element = g.find("div", class_=["VwiC3b", "yXK7lf", "MUxGbd", "yDYNvb", "lyLwlc"])
                if not snippet_element:
                    snippet_element = g.find("span", class_=["aCOpRe"])
                if not snippet_element:
                    snippet_element = g.find("div", class_="IsZvec")
                
                snippet = snippet_element.get_text() if snippet_element else "Không có mô tả"
                
                # Clean up text
                title = re.sub(r"\s+", " ", title).strip()
                snippet = re.sub(r"\s+", " ", snippet).strip()
                
                if len(snippet) > 300:
                    snippet = snippet[:300] + "..."
                
                results.append({
                    "title": title,
                    "link": link,
                    "snippet": snippet
                })
                
                if len(results) >= max_results:
                    break
                    
            except Exception as e:
                logger.debug(f"Error parsing result: {e}")
                continue
        
        if not results:
            logger.info("No structured results found, trying alternative parsing")
            # Alternative: try to extract any text content
            main_content = soup.find("div", {"id": "main"})
            if main_content:
                text = main_content.get_text(separator=" ", strip=True)
                # Extract first meaningful paragraph
                paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 50]
                if paragraphs:
                    results.append({
                        "title": "Kết quả tìm kiếm",
                        "link": "https://google.com/search?q=" + search_query,
                        "snippet": paragraphs[0][:300]
                    })
        
        if not results:
            logger.warning("No Google search results found")
            return None
        
        # Format results for AI
        formatted = f"Kết quả tìm kiếm Google cho: \"{query}\"\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   Nguồn: {result['link']}\n"
            formatted += f"   {result['snippet']}\n\n"
        
        logger.info(f"Google search found {len(results)} results")
        return formatted
        
    except Exception as e:
        logger.warning(f"Google search error: {e}")
        import traceback
        traceback.print_exc()
        return None


def reload_knowledge(force: bool = False):
    """Manually reload knowledge base."""
    global _knowledge_loaded
    _knowledge_loaded = False

    try:
        # Try both import paths for local vs deployment
        try:
            from backend.knowledge.retriever import retriever
        except ModuleNotFoundError:
            from knowledge.retriever import retriever

        retriever.load_knowledge(force_reload=force)
        logger.info("Knowledge base reloaded")
        return True
    except Exception as e:
        logger.error(f"Failed to reload knowledge: {e}")
        return False