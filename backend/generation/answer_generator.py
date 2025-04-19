import requests
import json
import tiktoken

class AnswerGenerator:
    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct"):
        self.model = model
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.api_key = "sk-vlcegmesrbdosyevomhivhfycryltocxwxtwwvwwnqtjqsxo"
    
    def create_prompt(self, question, contexts, generate_type):
        """创建提示"""
        # 格式化上下文
        formatted_contexts = ""
        for i, context in enumerate(contexts):
            formatted_contexts += f"文档片段 {i}:\n{context}\n\n"
        
        # 系统提示
        system_message = """
        You are a knowledge base question answering system. Please answer the user's questions according to the content of the provided documentation.
        Don't make up information that isn't in the documents. The answer should be concise and clear, and should be in English.
        """
        if generate_type == "user":
            # 创建用户消息
            user_message = f"""The following documents are available:
            {formatted_contexts}
            You need to answer all questions based on the documents above, and just give me the answer, don't give me any other information or your thoughts: 
            {question}
            **Important notes:**
                -The answer should only be the name of an entity (such as a person, place or movie name), a specific number or time(format like "Day(if have) Month(full month name if have) Year(if have)"). Please do not provide any other data as an answer.
                -If you can't answer the question base on the documents, just return the answer that you think is correct. But don't make up any information and the answer should be as brief as possible too.
                -If you can't answer the question that you think is correct, just return “Sorry, I don't know.”.
            """
        elif generate_type == "test":
            # 创建测试消息
            test_message = f"""The following documents are available:
            {formatted_contexts}
            You need to answer all questions based on the documents above: 
            {question}
            Please respond in the format {{"question": ...., "answer": ....}}, and the answer should be as brief as possible, the examples are as follow:
            Examples:
            {{"question": "when did the 1st world war officially end", "answer": "11 November 1918"}}
            {{"question": "name of the actor who plays captain america", "answer": "Christopher Robert Evans"}}
            {{"question": "when is the men's ice hockey winter olympics 2018", "answer": "between 14 and 25 February"}}
            
            **Important notes:**
                -The answer should only be the name of an entity (such as a person, place or movie name), a specific number or time(format like "Day(if have) Month(full month name if have) Year(if have)"). Please do not provide any other data as an answer.
                -If you can't answer the question base on the documents, just return the answer that you think is correct. But don't make up any information and the answer should be as brief as possible too.
                -If you can' answer the question that you think is correct, just return " ".
                -Insure that respond in the format {{"question": ...., "answer": ....}}, don't add any other extra information.
        """

        if generate_type == "test":
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": test_message}
            ]
        elif generate_type == "user":
            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
    
    def generate(self, question, contexts, generate_type="test"):
        """生成答案"""
        # 创建消息
        messages = self.create_prompt(question, contexts, generate_type)
        
        # 准备API请求
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 5,
            "response_format": {"type": "text"}
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # 发送API请求
            response = requests.post(self.api_url, json=payload, headers=headers)
            #print("response", response)
            
            # 解析响应
            response_data = response.json()
            #print("response_data", response_data)
            
            if "error" in response_data or "code" in response_data:
                return f"生成答案时出现错误: {json.dumps(response_data, ensure_ascii=False)}"
                
            answer = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return answer.strip()
            
        except Exception as e:
            print(f"API请求错误: {e}")
            return f"生成答案时出现错误，请稍后再试。错误: {str(e)}"


if __name__ == "__main__":
    answer_generator = AnswerGenerator()
    question = "What is the capital of France?"
    contexts = ["France is a country in Western Europe.", "Paris is the capital of France."]
    answer = answer_generator.generate(question, contexts, generate_type="test")
    print("answer: ", answer)

