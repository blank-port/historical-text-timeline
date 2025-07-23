const backendURL =
  window.location.hostname === "localhost"
    ? "http://127.0.0.1:8000"
    : "https://historical-text-timeline.onrender.com";


async function uploadPDF() {
    const fileInput = document.getElementById("pdf-file");
    const file = fileInput.files[0];
    const statusText = document.getElementById("upload-status");

    if (!file) {
        statusText.style.color = "red";
        statusText.textContent = "❌ Please select a PDF file.";
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    statusText.style.color = "blue";
    statusText.textContent = "⏳ Uploading and processing PDF...";

    try {
        const response = await fetch("http://127.0.0.1:8000/upload-pdf", {
            method: "POST",
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            statusText.style.color = "green";
            statusText.textContent = result.message || "✅ PDF processed successfully!";
        } else {
            statusText.style.color = "red";
            statusText.textContent = result.error || "❌ Failed to process PDF.";
        }
    } catch (error) {
        console.error("Error uploading PDF:", error);
        statusText.style.color = "red";
        statusText.textContent = "❌ Server error. See console for details.";
    }
}

async function askQuestion() {
    const question = document.getElementById("question").value;

    const formData = new FormData();
    formData.append("question", question);

    const response = await fetch(`${backendURL}/ask`, {
        method: "POST",
        body: formData,
    });

    const result = await response.json();
    document.getElementById("answer").innerText = result.answer || result.error;
}

async function showTimeline() {
    const response = await fetch(`${backendURL}/timeline`);
    const result = await response.json();

    const timelineContainer = document.getElementById("timeline");
    timelineContainer.innerHTML = "";

    if (result.timeline && Array.isArray(result.timeline)) {
        result.timeline.forEach(entry => {
            const section = document.createElement("div");
            section.innerHTML = `<h3>🗓 ${entry.year}</h3><ul>` +
                entry.events.map(event => `<li>${event}</li>`).join("") +
                "</ul><hr>";
            timelineContainer.appendChild(section);
        });
    } else {
        timelineContainer.innerText = result.timeline || result.error;
    }
}
