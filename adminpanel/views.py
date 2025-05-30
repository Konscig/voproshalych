from datetime import date, timedelta
from flask import render_template, redirect, request, url_for
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
import requests
from config import app
from cluster_analysis import ClusterAnalysis
from models import get_questions_count, get_questions_for_clusters, get_admins, Admin, get_chunks_count, get_documents_count, get_last_sync_date, get_unique_docs_urls
from datetime import datetime, timedelta


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "/login"
analysis = ClusterAnalysis()


@login_manager.user_loader
def load_user(id):
    return Admin.query.get(int(id))


@app.route("/login", methods=["GET", "POST"])
def login():
    """Функция авторизует пользователя, если данные для входа совпадают

    Returns:
        str: отрендеренная главная веб-страница сервиса
    """

    if current_user.is_authenticated:
        return redirect(url_for("index"))
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        admin = Admin.query.filter_by(email=email).first()
        if admin and admin.check_password(password):
            login_user(admin)
            return redirect(url_for("index"))
        else:
            status = "Авторизация отклонена."
            return render_template("login.html", status=status)
    else:
        return render_template("login.html")


@app.post("/logout")
@login_required
def logout():
    """Функция деавторизует пользователя

    Returns:
        str: отрендеренная веб-страница авторизации
    """
    logout_user()
    return redirect(url_for("login"))


@app.get("/")
@login_required
def index() -> str:
    """Функция рендерит главную страницу веб-сервиса

    Returns:
        str: отрендеренная главная веб-страница
    """
    time_start = str(date.today() - timedelta(days=30))
    time_end = str(date.today() + timedelta(days=1))
    question_counts = get_questions_count(time_start=time_start, time_end=time_end)
    question_counts_lists = (
        list(question_counts.keys()),
        [i[0] for i in question_counts.values()],
        [i[1] for i in question_counts.values()],
    )
    return render_template(
        "main-page.html",
        question_counts=question_counts_lists,
        page_title="Сводка",
    )


@app.get("/questions-analysis")
@login_required
def questions_analysis() -> str:
    """Функция выводит на экране вопросы, не имеющие ответа

    Returns:
        str: отрендеренная веб-страница с POST-запросом на базу данных
    """
    if len(request.values.keys()) == 0:
        time_start = str(date.today() - timedelta(days=30))
        time_end = str(date.today() + timedelta(days=1))
        have_not_answer = True
        have_low_score = False
        have_high_score = False
        have_not_score = False
    else:
        time_start = str(request.values.get("time_start"))
        time_end = str(request.values.get("time_end"))
        have_not_answer = bool(request.values.get("have_not_answer"))
        have_low_score = bool(request.values.get("have_low_score"))
        have_high_score = bool(request.values.get("have_high_score"))
        have_not_score = bool(request.values.get("have_not_score"))
    questions = get_questions_for_clusters(
        time_start=time_start,
        time_end=time_end,
        have_not_answer=have_not_answer,
        have_low_score=have_low_score,
        have_high_score=have_high_score,
        have_not_score=have_not_score,
    )
    marks = {
        "нет оценки": "✏️",
        "высокая оценка": "✔️",
        "низкая оценка": "❌",
        "нет ответа": "❓",
    }
    clusters, questions_len, clusters_len = analysis.get_clusters_keywords(questions)
    return render_template(
        "questions-analysis.html",
        time_start=time_start,
        time_end=time_end,
        have_not_answer=have_not_answer,
        have_low_score=have_low_score,
        have_high_score=have_high_score,
        have_not_score=have_not_score,
        clusters=clusters,
        questions_len=questions_len,
        clusters_len=clusters_len,
        marks=marks,
        page_title="Анализ вопросов",
    )


@app.route("/broadcast", methods=["POST", "GET"])
@login_required
def broadcast() -> str:
    """Функция отправляет HTML-POST запрос на выполнение массовой рассылки на HOST чат-бота

    Returns:
        str: отрендеренная веб-страница с POST-запросом на сервер
    """

    if request.method == "POST":
        text = request.form.get("name")
        vk_bool = bool(request.form.get("vk"))
        tg_bool = bool(request.form.get("tg"))
        response = requests.post(
            url=f"http://{app.config['CHATBOT_HOST']}/broadcast/",
            json={"text": text, "tg": tg_bool, "vk": vk_bool},
        )
        return render_template(
            "broadcast.html", page_title="Рассылка", response=response.text
        )
    return render_template("broadcast.html", page_title="Рассылка")


@app.get("/settings")
@login_required
def settings() -> str:
    """Функция выводит интерфейс взаимодействия с администраторами панели и с непосредственно модулем QA

    Returns:
        str: отрендеренная веб-страница настроек администраторов и возможностью провести переиндексацию
    """
    users = get_admins()

    return render_template("settings.html", users=users, page_title="Настройки")


@app.post("/reindex")
@login_required
def reindex_qa():
    """Функция отправляет POST-запрос на переиндексацию в модуле QA

    Returns:
        str: статус отправки запроса
    """
    requests.post(f"http://{app.config['QA_HOST']}/reindex/", timeout=1000)
    return redirect(url_for("settings"))


@app.post("/reembed")
@login_required
def reembed_qa():
    """Функция отправляет POST-запрос на пересоздание векторных представлений вопросов в модуле QA

    Returns:
        str: статус отправки запроса
    """
    requests.post(f"http://{app.config['QA_HOST']}/reembed/", timeout=1000)
    return redirect(url_for("settings"))


@app.post("/check_score")
@login_required
def check_scores_qa():
    """Функция отправляет POST-запрос на пересоздание векторных представлений вопросов в модуле QA

    Returns:
        str: статус отправки запроса
    """
    requests.post(f"http://{app.config['QA_HOST']}/check-score/", timeout=1000)
    return redirect(url_for("settings"))




@app.get("/service_stats")
@login_required
def service_stats() -> str:
    """
    Description:

    Функция выводит статистику сервиса

    Returns:
    str: отрендеренная веб-страница с POST-запросом на сервер
    """

    if len(request.values.keys()) == 0:
        time_start = str(date.today() - timedelta(days=30))
        time_end = str(date.today() + timedelta(days=1))
    else:
        time_start = str(request.values.get("time_start"))
        time_end = str(request.values.get("time_end"))

    x_questions = get_questions_for_clusters(
        time_start=time_start,
        time_end=time_end,
        have_not_answer=False,
        have_low_score=True,
        have_high_score=False,
        have_not_score=False,
    )

    v_questions = get_questions_for_clusters(
        time_start=time_start,
        time_end=time_end,
        have_not_answer=False,
        have_low_score=False,
        have_high_score=True,
        have_not_score=False,
    )

    x_count = len(x_questions)
    v_count = len(v_questions)
    total = x_count + v_count
    documents_count = get_documents_count()
    chunks_count = get_chunks_count()
    sync_date = get_last_sync_date()
    documents = get_unique_docs_urls()

    return render_template(
        'service_stats.html',
        page_title='Статистика сервиса',
        documents=documents,
        documents_count=documents_count,
        chunks_count=chunks_count,
        sync_date=sync_date,
        x_count=x_count,
        v_count=v_count,
        x_questions=x_questions,
        v_questions=v_questions,
        time_start=time_start,
        time_end=time_end,
        question_counts=total
    )
