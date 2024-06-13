import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc,mean_absolute_error,classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
import os
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import plot_tree
        
# アプリケーションのタイトル
st.write("電卓ネコシリーズ第２弾！")
st.title("ネコでも使える！会計AI分析")

title = "nekoai_title.png" 
image = Image.open(title)
st.image(image,use_column_width=True)

st.write("青山学院大学矢澤研究室では、人工知能（AI）を用いた会計・財務分析に取り組んでいます。ウェブアプリ「ネコでも使える！会計AI分析（略して、ネコAI）」は、機械学習を用いた売上・株価分析をクリックだけで実行できます。肩の力を抜いてお楽しみください！")
st.write("ご利用にあたっては、ページ最下段の【諸注意】と【免責事項】もお読みください。")
st.write("最終更新日：2024年X月XX日")

st.title("はじめに")
st.write("機械学習とは！？")
st.write("機械学習とは、「データから知識を引き出す」ことをいいます。機械学習には「教師あり学習」に加えて、「教師なし学習」「強化学習」があります。このアプリでは、教師あり学習による分類・回帰モデルの学習とモデルを使った予測を行うことができます。本当はアルゴリズムの仕組みやプログラミング言語の習得などちょっと面倒な作業があるのですが、ここではクリックしていくつか設定するだけ機械学習モデルが構築できます。")
st.write("それではサイドバーに表示されているステップ１から進めていきましょう！")

st.sidebar.title("ステップ１：データの収集")
st.sidebar.write("学習用のデータをアップロードしてください。")
# ファイルアップローダーの設置
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
st.sidebar.write("サンプルデータを下記からダウンロードできます。")
# ファイルパス
file_path = "sample_data.csv"
# ファイルが存在するかチェック
if os.path.exists(file_path):
    # ダウンロードボタン
    with open(file_path, "rb") as file:
        st.sidebar.download_button(
            label="サンプルデータ",
            data=file,
            file_name="sample_data.csv",
            mime="text/csv"
        )
else:
    st.error("ファイルが見つかりません。パスを確認してください。")
st.sidebar.write("※東証プライム上場、2022年3月期決算、2020年3月期から2023年3月期のデータが入手可能な会社")

# トレーニング済みモデルと結果をセッション状態で管理
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
    st.session_state['training_results'] = None
    st.session_state['test_data'] = None 

problem_type = ''

st.title("ステップ２：データの観察と前処理")
st.write("データをよく観察するとともに、欠損値の処理、外れ値の検出と対処、場合に応じてデータの正規化などの前処理を行います")

if uploaded_file is not None:
    # CSVファイルの読み込み
    df = pd.read_csv(uploaded_file)

    # 数値列のみを抽出
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # 記述統計量を計算
    description = numeric_df.describe(percentiles=[0.1, 0.5, 0.9]).T

    # 必要な統計量を選択
    description = description[['count', 'mean', 'std', '90%', '50%', '10%']]
    # 列名の変更
    description.columns = ['N', '平均', '標準偏差', '90%', '中央値', '10%']

    # 数値を小数点第2位まで丸める
    description = description.round(2)
    
    # 結果の表示
    st.write('記述統計：')
    st.dataframe(description)

    # 相関係数表を計算
    correlation_matrix = numeric_df.corr()

    # 相関係数表を小数点第2位まで丸める
    correlation_matrix = correlation_matrix.round(2)

    # 相関係数表の表示
    #st.write('相関係数表：')
    #st.dataframe(correlation_matrix)
        
    # ヒートマップの表示
    st.write('相関係数（ヒートマップ：')
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=False, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)

st.title("ステップ３：特徴量の選択")
st.write("教師データと特徴量を選択します。その後、モデルを選択してモデルのトレーニングを行います。")

if uploaded_file is not None:
    # アップロードされたファイルを読み込む
    #data = pd.read_csv(uploaded_file)
    data = df

    # 教師データの選択
    label = st.selectbox("教師データを選択してください", options=data.columns)

    # 選択された列のユニークな値を取得
    unique_values = data[label].unique()

    # 特徴量の選択
    features = st.multiselect("特徴量を選択してください", options=[col for col in data.columns if col != label])

    # 選択された特徴量の数をチェック
    if len(features) > 50:
        st.error("特徴量は最大で50個まで入力できます。選択された特徴量の数を減らしてください。")
        # 選択された特徴量を50個に制限する（ユーザーが多く選択した場合の対応例）
        features = features[:50]
    else:
        features = features

    # 特徴量が選択された後に実行
    if features:
        # 選択された特徴量のデータタイプを検証
        non_numeric_features = [feature for feature in features if data[feature].dtype not in ['int64', 'float64']]

        # 非数値特徴量が存在する場合、警告を表示
        if non_numeric_features:
            st.warning("以下の特徴量は数値形式ではありません。数値に変換するか、除外してください：" + ", ".join(non_numeric_features))
            # ここで数値に変換する処理を促すか、自動で変換するコードを追加することもできます。

st.title("ステップ４：学習")
st.write("①学習の設定")

if 'features' in locals() or 'features' in globals():
    if features:
        # 問題のタイプを選択
        problem_type = st.selectbox("問題のタイプを選択してください", ['分類', '回帰'])

        st.session_state.problem_type = problem_type  # セッション状態に問題のタイプを保存

        # アルゴリズムを選択
        if problem_type == '分類':
            if not all(value in [0, 1] for value in unique_values):
                st.error("分類タスクの教師データは離散変数（0、1）を入れてください。")
            model_type = st.selectbox("アルゴリズムを選択してください", [
                'K-最近傍法',
                'ロジスティック回帰',
                '決定木',
                'ランダムフォレスト', 
                '勾配ブースティング', 
            ])
            st.write('ハイパーパラメータ（そのままでも可）')
            if model_type == 'K-最近傍法':
                n_neighbors = st.number_input("近傍点の数を設定してください（1-10）", min_value=1, max_value=10, value=5)
            elif model_type == 'ロジスティック回帰':
                C = st.slider("正則化の強さ(C)を設定してください（0.01-1.0）", min_value=0.01, max_value=1.0, value=0.01, step=0.01)
            elif model_type == '決定木':
                max_depth = st.number_input("決定木の最大深さを設定してください（1-5）", min_value=1, max_value=5,value=3)
            elif model_type == 'ランダムフォレスト':
                n_estimators = st.number_input("決定木の数を設定してください（10-100）", min_value=10, max_value=100, value=50)
            elif model_type == '勾配ブースティング':
                n_estimators = st.number_input("決定木の数を設定してください（10-100）", min_value=10, max_value=100,value=50)
                
        elif problem_type == '回帰':
            label_dtype = data[label].dtype
            if data[label].dtype in ['int64', 'float64'] and all(data[label] % 1 == 0) and data[label].between(0, 30).all():
                st.warning("回帰タスクの教師データには、離散変数（カテゴリ）ではなく連続変数を選択してください。")
            
            model_type = st.selectbox("アルゴリズムを選択してください", [
                'K-最近傍法',
                '線形回帰',
                '決定木',
                'ランダムフォレスト',
                '勾配ブースティング',
            ])
            
        # モデルタイプをセッション状態に保存
        st.session_state.model_type = model_type

st.write("②モデルのトレーニング")
if 'features' in locals() or 'features' in globals():
    if features:
        if st.button("モデルをトレーニングする") and label and features and model_type:
            # 特徴量とラベルの定義
            X = data[features]
            y = data[label]
            
            # 訓練データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 分割したデータをセッション状態に保存
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            # 分類モデルの初期化とトレーニング
            if problem_type == '分類':
                if model_type == 'K-最近傍法':
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)  # n_neighborsを動的に設定
                elif model_type == 'ロジスティック回帰':
                    model = LogisticRegression(C=C, random_state=42)  # Cを動的に設定
                elif model_type == '決定木':
                    from sklearn.tree import DecisionTreeClassifier
                    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)  # max_depthを既に動的に設定
                elif model_type == 'ランダムフォレスト':
                    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)  # n_estimatorsを動的に設定
                elif model_type == '勾配ブースティング':
                    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)  # n_estimatorsを動的に設定

                # トレーニング
                model.fit(X_train, y_train)
                st.success("トレーニングが成功しました！")
                
                # トレーニング結果をセッション状態に保存
                st.session_state.model_trained = True
                st.session_state.model = model  # トレーニングしたモデルをセッション状態に保存
                st.session_state.features = features  # 特徴量もセッション状態に保存
                
            # 回帰モデルの初期化とトレーニング
            elif problem_type == '回帰':
                if model_type == 'ランダムフォレスト':
                    model = RandomForestRegressor(n_estimators=50,random_state=42)
                elif model_type == '決定木':
                    from sklearn.tree import DecisionTreeRegressor
                    model = DecisionTreeRegressor(max_depth=3,random_state=42)
                elif model_type == '線形回帰':
                    model = LinearRegression()
                elif model_type == '勾配ブースティング':
                    model = GradientBoostingRegressor(n_estimators=50, random_state=42)
                elif model_type == 'K-最近傍法':
                    model = KNeighborsRegressor(n_neighbors=5)
                # トレーニング
                model.fit(X_train, y_train)
                st.success("トレーニングが成功しました！")

                # トレーニング結果をセッション状態に保存
                st.session_state.model_trained = True
                st.session_state.model = model  # トレーニングしたモデルをセッション状態に保存
                st.session_state.features = features  # 特徴量もセッション状態に保存

st.title("ステップ５：評価")
st.write("トレーニングが完了したモデルをテストデータに適用して精度を評価します。")

# 確率に基づくメッセージを返す関数
def interpret_probability(probability):
    if probability <= 0.10:
        return f"予測確率は{probability:.2f}です。ほとんどあり得ないでしょう。"
    elif probability <= 0.20:
        return f"予測確率は{probability:.2f}です。かなり低いでしょう。"
    elif probability <= 0.30:
        return f"予測確率は{probability:.2f}です。低いでしょう。"
    elif probability <= 0.40:
        return f"予測確率は{probability:.2f}です。やや低いでしょう。"
    elif probability <= 0.50:
        return f"予測確率は{probability:.2f}です。どちらともいえないでしょう。"
    elif probability <= 0.60:
        return f"予測確率は{probability:.2f}です。やや確率は高いでしょう。"
    elif probability <= 0.70:
        return f"予測確率は{probability:.2f}です。確率は高いでしょう。"
    elif probability <= 0.80:
        return f"予測確率は{probability:.2f}です。割と確率は高いでしょう。"
    elif probability <= 0.90:
        return f"予測確率は{probability:.2f}です。非常に確率は高いでしょう。"
    else:
        return f"予測確率は{probability:.2f}です。ほぼ確実でしょう。"

if problem_type == '分類' and 'model_trained' in st.session_state and st.session_state.model_trained:
    model = st.session_state.model # トレーニング済みモデルを取得
    X_test = st.session_state.X_test  # X_testをセッション状態から取得
    y_test = st.session_state.y_test  # y_testをセッション状態から取得
    X_train = st.session_state.X_train  # X_trainをセッション状態から取得
    y_train = st.session_state.y_train  # y_trainをセッション状態から取得
    y_pred = model.predict(X_test)

    # 精度の算出
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f'モデルの正解率（Accuracy）: {accuracy:.2f}')

    # ROC曲線とAUCの表示
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        st.write("AUCスコアとROC曲線:")
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')  # 50% AUCの線
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('偽陽性率（False Positive Rate）')
        plt.ylabel('真陽性率（True Positive Rate）')
        plt.title('ROC（Receiver Operating Characteristic）')
        plt.legend(loc="lower right")
        st.pyplot(plt)

    # 決定木のツリー構造を出力
    if model_type == '決定木':
        plt.figure(figsize=(20,10))
        plot_tree(model, filled=True, feature_names=features, class_names=True)
        st.pyplot(plt)

    # Feature Importanceの表示
    if hasattr(model, "feature_importances_"):
        st.write("特徴量の重要度 (Feature Importance):")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # 重要度が高い特徴量のインデックスを降順で取得

        plt.figure(figsize=(10, 8))  # グラフのサイズを設定
        plt.title("特徴量の重要度 (Feature Importance)")
        plt.barh(range(X_train.shape[1]), importances[indices], color="r", align="center")  # 水平棒グラフを描画
        plt.yticks(range(X_train.shape[1]), np.array(features)[indices])  # y軸に特徴量の名前を設定
        plt.gca().invert_yaxis()  # y軸の順序を逆にして、重要度が高い特徴量を上に表示
        plt.xlabel("Importance")  # x軸のラベル
        plt.ylabel("Features")  # y軸のラベル
        st.pyplot(plt)
                
if problem_type == '回帰' and 'model_trained' in st.session_state and st.session_state.model_trained:
    model = st.session_state.model # トレーニング済みモデルを取得
    X_test = st.session_state.X_test  # X_testをセッション状態から取得
    y_test = st.session_state.y_test  # y_testをセッション状態から取得
    X_train = st.session_state.X_train  # X_trainをセッション状態から取得
    y_train = st.session_state.y_train  # y_trainをセッション状態から取得
    # 回帰問題の性能評価（例：MSE、R²）
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) *100
    # MAEの計算
    mae = mean_absolute_error(y_test, y_pred)
    # MAPEの計算
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    st.write(f"MAE (平均絶対誤差): {mae:.2f}")
    #st.write(f"MAPE (平均絶対パーセンテージ誤差): {mape:.2f}%")
    st.write(f"R2（決定係数）: {r2:.2f}%")  # R2をパーセントで表示
    # 予測値と実際の値の散布図
    y_pred = model.predict(X_test)
    plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel('実際の値')
    plt.ylabel('予測値')
    plt.title('予測値 vs 実際の値')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # 完全一致の線
    st.pyplot(plt)

    # Feature Importanceの表示
    if hasattr(model, "feature_importances_"):
        st.write("特徴量の重要度 (Feature Importance):")
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]  # 重要度が高い特徴量のインデックスを降順で取得

        plt.figure(figsize=(10, 8))  # グラフのサイズを設定
        plt.title("特徴量の重要度 (Feature Importance)")
        plt.barh(range(X_train.shape[1]), importances[indices], color="r", align="center")  # 水平棒グラフを描画
        plt.yticks(range(X_train.shape[1]), np.array(features)[indices])  # y軸に特徴量の名前を設定
        plt.gca().invert_yaxis()  # y軸の順序を逆にして、重要度が高い特徴量を上に表示
        plt.xlabel("Importance")  # x軸のラベル
        plt.ylabel("Features")  # y軸のラベル
        st.pyplot(plt)

# Step3 トレーニングしたモデルを使って予測する
st.title("予測")
st.write("トレーニングしたモデルを用いて予測を行います。")

if 'model_trained' in st.session_state and st.session_state.model_trained:
    #　問題のタイプを表示
    st.write("問題のタイプ:",st.session_state.problem_type)
    # トレーニングモデルのタイプを表示
    st.write("トレーニングモデル:", st.session_state.model_type)
    
    # 予測用の入力値を受け取る
    input_data = []
    for feature in st.session_state.features:
        value = st.number_input(f"{feature}の値を入力してください", key=feature)
        input_data.append(value)

    # 予測の実行
    if st.button("モデルを使って予測する"):
        model = st.session_state['model']  # セッション状態からモデルを取り出す
        input_data_np = np.array(input_data).reshape(1, -1)  # 入力データをNumPy配列に変換
        
        # モデルが確率を提供する場合はpredict_probaを使用、それ以外はpredictを使用
        if hasattr(model, "predict_proba"):
            prediction = model.predict_proba(input_data_np)[0, 1]
            message = interpret_probability(prediction)
            st.write(message)
        else:
            prediction = model.predict(input_data_np)
            st.write(f"予測結果: {prediction[0]:.2f}")

st.title('おわりに')
st.write("AI分析はいかがでしたでしょうか？次はぜひ自分の興味のあるデータを入れてモデルをつくってみましょう！")

st.write("【サイト運営者】")
st.write("青山学院大学　経営学部　矢澤憲一研究室")
st.write("【諸注意】")
st.write("１．私的目的での利用について：")
st.write("本ウェブアプリケーションは、個人的な用途で自由にご利用いただけます。しかしながら、公序良俗に反する行為は固く禁じられています。利用者の皆様には、社会的な規範を尊重し、責任ある行動をお願いいたします。")
st.write("２．ビジネス目的での利用について：")
st.write("本アプリケーションをビジネス目的で使用される場合は、事前に以下の連絡先までご一報ください。使用に関する詳細な情報を提供いたします。")
st.write("２．学術論文執筆目的での利用について：")
st.write("学術論文の執筆に当たり、本アプリケーションのデータや機能を利用される場合は、下記の文献を参考文献として明記し、同時に以下の連絡先までご一報いただくようお願いいたします。")
st.write("参考文献：執筆中")
st.write("連絡先：yazawa(at)busi.aoyama.ac.jp")
st.write("【免責事項】")
st.write("このウェブサイトおよびそのコンテンツは、一般的な情報提供を目的としています。このウェブサイトの情報を使用または適用することによって生じるいかなる利益、損失、損害について、当ウェブサイトおよびその運営者は一切の責任を負いません。情報の正確性、完全性、時宜性、適切性についても、一切保証するものではありません。")
