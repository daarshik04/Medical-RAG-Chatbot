# FIX 5: create_qa_chain() was called on every POST request — reloading the
#         vector store + embedding model each time. The chain is now built once
#         at startup via the cached get_qa_chain() inside retriever.py.
#
# FIX 6: The old chain expected {"query": user_input} (RetrievalQA convention).
#         The new LCEL chain takes a plain string: chain.invoke(user_input).
#         Response is now a plain string — no more .get("result") needed.
#
# FIX 10: load_dotenv() was called twice. Removed the duplicate call.

from flask import Flask, render_template, request, session, redirect, url_for
from markupsafe import Markup
from dotenv import load_dotenv
import os

load_dotenv()  # single call — FIX 10

from app.components.retriever import get_qa_chain

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))


# @app.template_filter("nl2br")
# def nl2br(value: str) -> Markup:
#     return Markup(value.replace("\n", "<br>\n"))


@app.route("/", methods=["GET", "POST"])
def index():
    if "messages" not in session:
        session["messages"] = []

    if request.method == "POST":
        user_input = request.form.get("prompt", "").strip()

        if user_input:
            messages = session["messages"]
            messages.append({"role": "user", "content": user_input})
            session["messages"] = messages

            try:
                # FIX 5: get_qa_chain() returns the cached chain — no reload
                chain = get_qa_chain()
                if chain is None:
                    raise Exception("QA chain could not be created (check LLM and vector store).")

                # FIX 6: LCEL chain takes a plain string, returns a plain string
                result = chain.invoke(user_input)

                messages.append({"role": "assistant", "content": result})
                session["messages"] = messages

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                return render_template(
                    "index.html",
                    messages=session["messages"],
                    error=error_msg,
                )

        return redirect(url_for("index"))

    return render_template("index.html", messages=session.get("messages", []))


@app.route("/clear")
def clear():
    session.pop("messages", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
