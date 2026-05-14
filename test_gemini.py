import google.generativeai as genai

genai.configure(
    api_key="AIzaSyCbvbys4zHhz-OgtaqyEsG3f8SVDZdM5jQ"
)

for model in genai.list_models():
    print(model.name)