import os
import sys
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def load_data(csv_path):
    """Загружаем CSV и возвращаем DataFrame."""
    return pd.read_csv(csv_path)

def setup_model():
    """Настраиваем LLaMA 3 с API-ключом."""
    os.environ["GROQ_API_KEY"] = "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"
    return ChatGroq(model="llama3-70b-8192")  # Можно заменить на 8B

def create_prompt():
    """Создаём промпт для выбора ответа."""
    template = '''Ты искусственный интеллект, который решает тесты.
    Вопрос: {question}
    Варианты ответов: {choices}
    Ответь только числом (0-3), указывая правильный вариант.'''
    return PromptTemplate(input_variables=['question', 'choices'], template=template)

def process_questions(df, llm):
    """Обрабатываем вопросы и получаем ответы."""
    prompt = create_prompt()
    chain = LLMChain(llm=llm, prompt=prompt, output_key='answer')
    
    answers = []
    for _, row in df.iterrows():
        response = chain.invoke({
            'question': row['question'],
            'choices': row['choices']
        })['answer']
        
        try:
            answer = int(response.strip())
        except ValueError:
            answer = 1  # Если модель ошиблась
        
        answers.append({'id': row['Номер вопроса'], 'answer': answer})
    
    return pd.DataFrame(answers)

def main():
    """Главная функция для запуска скрипта."""
    if len(sys.argv) < 3:
        print("Использование: python main.py input.csv output.csv")
        sys.exit(1)
    
    input_csv, output_csv = sys.argv[1], sys.argv[2]
    df = load_data(input_csv)
    llm = setup_model()
    results = process_questions(df, llm)
    results.to_csv(output_csv, index=False)
    print(f"✅ Ответы сохранены в {output_csv}")

if __name__ == "__main__":
    main()
