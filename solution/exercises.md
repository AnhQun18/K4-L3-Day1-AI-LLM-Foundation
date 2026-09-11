# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

prompt = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."
temperatures = [0.0, 0.5, 1.0, 1.5]

for temp in temperatures:
    print(f"\n[ Temperature = {temp} ]")
    # Gọi hàm gọi API
    response_text, latency = call_openai(prompt, temperature=temp)
    print(f"- Thời gian: {latency:.4f} giây")
    print(f"- Kết quả:\n{response_text}")


**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Temperature càng cao, câu trả lời của AI càng khó xác minh, dễ bị hallucination

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Mình sẽ chọn temperature khoảng 0.3
  Lí do: Ở mức temp xấp xỉ 0.7, chatbot dễ rơi vào tình trạng hallucination
         Chatbot CSKH cần lấy dữ liệu sát với tài liệu nội bộ của doanh nghiệp
         Không cần lấy mức quá thấp vì đôi khi sẽ cần lời văn để thuyết phục khách hàng

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> GPT-4o đắt hơn GPT-4o-mini khoảng 16.7 lần 
  GPT-4o xứng đáng cho các tác vụ suy luận đa bước phức tạp, viết code chuyên sâu hoặc phân tích pháp lý/tài chính đòi hỏi độ chính xác tuyệt đối.
---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> System prompt hoạt động như một "bộ lọc" để giới hạn không gian xử lý của AI, quyết định 3 yếu tố: từ vựng, giọng điệu, độ sâu. với prompt 1, chatbot nói ngắn gọn, từ ngữ đời thường. với prompt 2, câu trả lời chi tiết hơn, dày đặc thuật ngữ và cấu trúc học thuật

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn 113 từ, công thức 'số từ / 0.75' ước lượng ra 151 token, trong khi tiktoken của GPT-4o (o200k_base) cho ra 137 token (chênh lệch ~10%), còn trên GPT-4 cũ (cl100k_base) lên tới 246 token (chênh lệch ~63%). Tiếng Việt tốn nhiều token hơn tiếng Anh vì: (1) Ngữ liệu huấn luyện tokenizer thiên lệch tiếng Anh nên từ điển thiếu các từ tiếng Việt nguyên khối; (2) Ký tự có dấu chiếm nhiều byte trong UTF-8 khiến thuật toán BPE phải tách nhỏ từ thành nhiều subword/byte tokens; (3) Cấu trúc từ ghép rời rạc có khoảng trắng làm tăng thêm số đơn vị phân đoạn.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng giao tiếp trực tiếp với người dùng (như chatbot hoặc trình soạn thảo tương tác) nhằm tối ưu độ trễ cảm nhận (perceived latency) và giảm thời gian chờ token đầu tiên (TTFT), giúp người dùng đọc được nội dung ngay khi sinh ra và có thể dừng sớm nếu cần. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm (background batch jobs, phân tích dữ liệu hàng loạt) hoặc khi cần sinh đầu ra có cấu trúc (JSON mode, function calling) vì hệ thống backend bắt buộc phải đợi nhận đủ toàn bộ chuỗi mới có thể parse dữ liệu và thực thi logic tiếp theo, đồng thời giúp kiến trúc mạng đơn giản hơn mà không cần duy trì kết nối SSE kéo dài.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> "So với delay cố định, exponential backoff tự động kéo dài thời gian chờ theo cấp số nhân sau mỗi lần thử, giúp giảm nhanh mật độ request để server có 'khoảng thở' phục hồi và reset rate-limit window. Nếu hàng nghìn client cùng retry với delay cố định (ví dụ 1s), chúng sẽ bị đồng bộ hóa và đồng loạt gửi request lại tại cùng một thời điểm, gây ra hiện tượng Thundering Herd (bão retry / retry storm). Đợt sóng truy vấn dồn dập này sẽ liên tục làm server vừa hồi phục lại bị quá tải ngay lập tức, dẫn đến sập dây chuyền kéo dài."

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là 'Trợ giảng khóa AI thân thiện'. System prompt: 'Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt.' Hai lựa chọn từ ngữ quan trọng: (1) 'Trả lời ngắn gọn' nhằm hạn chế tính dài dòng mặc định của LLM, giúp tiết kiệm chi phí output token (vốn đắt gấp 4 lần input) và tối ưu độ trễ hiển thị trên CLI; (2) 'Bằng tiếng Việt' để đảm bảo câu trả lời luôn đồng nhất bằng tiếng Việt ngay cả khi người dùng hỏi các câu có chứa nhiều thuật ngữ kỹ thuật tiếng Anh (như latency, streaming, temperature)

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là cơ chế cắt history chỉ giữ 3 lượt gần nhất qua history[-6:].
 Đề xuất cải thiện: Triển khai 'Conversation Summary Buffer' (Tóm tắt ngữ cảnh tích lũy). Cách triển khai: Khi hội thoại vượt quá 3 lượt, thay vì xóa bỏ, ta dùng GPT-4o-mini để tóm tắt các lượt chat cũ thành một đoạn văn ngắn gọn. Nhờ đó, bot duy trì được trí nhớ dài hạn mà không làm bùng nổ số lượng token.
---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
