const messagesEl = document.querySelector("#messages");
const form = document.querySelector("#composer");
const input = document.querySelector("#input");
const send = document.querySelector("#send");
const clear = document.querySelector("#clear");
const prompts = document.querySelectorAll(".prompt");

let messages = [
  {
    role: "assistant",
    content:
      "Ask me about The Odyssey. I am a small Qwen model fine-tuned on the poem, so keep questions close to Odysseus, Ithaca, Penelope, the gods, the sea, and the return home.",
  },
];

function render() {
  messagesEl.innerHTML = "";
  for (const message of messages) {
    const row = document.createElement("div");
    row.className = `message ${message.role}`;

    const role = document.createElement("div");
    role.className = "role";
    role.textContent = message.role === "user" ? "You" : "SLM";

    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = message.content;

    row.append(role, bubble);
    messagesEl.append(row);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addPending() {
  const row = document.createElement("div");
  row.className = "message assistant pending";
  row.id = "pending";

  const role = document.createElement("div");
  role.className = "role";
  role.textContent = "SLM";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = "Thinking through the poem...";

  row.append(role, bubble);
  messagesEl.append(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function removePending() {
  document.querySelector("#pending")?.remove();
}

function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 180)}px`;
}

async function submitMessage(text) {
  const content = text.trim();
  if (!content || send.disabled) return;

  messages.push({ role: "user", content });
  input.value = "";
  resizeInput();
  render();
  addPending();
  send.disabled = true;

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: messages.filter((message) => message.role !== "system") }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Chat request failed");
    messages.push({ role: "assistant", content: payload.reply || "(empty response)" });
  } catch (error) {
    messages.push({ role: "assistant", content: `Server error: ${error.message}` });
  } finally {
    removePending();
    send.disabled = false;
    render();
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitMessage(input.value);
});

input.addEventListener("input", resizeInput);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitMessage(input.value);
  }
});

clear.addEventListener("click", () => {
  messages = messages.slice(0, 1);
  render();
  input.focus();
});

for (const button of prompts) {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt || "";
    resizeInput();
    input.focus();
  });
}

render();
resizeInput();
