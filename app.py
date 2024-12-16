from flask import Flask, request, jsonify, render_template
from RAG import RAG1

app = Flask(__name__)

# 新接口函數：輸入字典，輸出字典
def function1(input_dict):
    print(input_dict)
    result = {}
    for cluster, markers_str in input_dict.items():
        markers = [m.strip() for m in markers_str.split(",")]
        result[cluster] = f"Function1 processed {len(markers)} markers"
    return result

def function2(input_dict):
    result = {}
    for cluster, markers_str in input_dict.items():
        markers = [m.strip() for m in markers_str.split(",")]
        result[cluster] = f"Function2 combined markers: {', '.join(markers)}"
    return result

def function3(input_dict):
    result = {}
    for cluster, markers_str in input_dict.items():
        markers = [m.strip() for m in markers_str.split(",")]
        result[cluster] = f"Function3 marker count: {len(markers)}"
    return result

# 定義函式對應的映射
FUNCTION_MAPPING = {
    "Function1": RAG1,
    "Function2": function2,
    "Function3": function3
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        # 從前端取得 JSON 資料
        data = request.json
        selected_function = data.get("function")
        clusters = data.get("clusters")  # 輸入字典 {"cluster1": "A, B, C", "cluster2": "D, E, F"}

        if not clusters or not isinstance(clusters, dict):
            return jsonify({"error": "Invalid input format. Expected a dictionary."}), 400

        # 調用對應函數並返回結果
        if selected_function in FUNCTION_MAPPING:
            result = FUNCTION_MAPPING[selected_function](clusters)
            return jsonify(result)
        else:
            return jsonify({"error": "Invalid function selected"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
