// Mapeo de palabras clave a URLs de videos
const videos = {
    "hola": "../videos/hola.mp4",  // Cambié las palabras clave a minúsculas
    "gracias": "../videos/gracias.mp4"
};

// Obtener elementos HTML
const keywordInput = document.getElementById("keywordInput");
const videoPlayer = document.getElementById("videoPlayer");
const videoElement = document.getElementById("video");
const videoSource = document.getElementById("videoSource");

// Función para actualizar el video
function playVideo(keyword) {
    // Buscar si la palabra clave tiene un video asociado
    if (videos[keyword]) {
        videoSource.src = videos[keyword];
        videoElement.load();
        videoElement.style.display = "block";
        videoElement.play();
        videoElement.loop = true;  // Reproduce en loop
        videoElement.muted = true; // Mute el video
    } else {
        // Ocultar el video si la palabra clave no coincide
        videoElement.style.display = "none";
        videoElement.loop = false; // Deshabilitar loop cuando no coincide
        videoElement.muted = false; // Desactivar mute cuando no hay video
    }
}

// Escuchar los cambios en el área de texto
keywordInput.addEventListener("input", () => {
    const inputText = keywordInput.value.trim().toLowerCase(); // Convertir a minúsculas
    playVideo(inputText);
});

// Escuchar cuando el usuario presiona una tecla
keywordInput.addEventListener("keydown", (event) => {
    // Verificar si la tecla presionada es "Backspace" (código 8)
    if (event.key === "Backspace") {
        keywordInput.value = ""; // Borrar toda la palabra
        playVideo(""); // Detener el video
    }
});

// Obtener el botón Home
const homeButton = document.getElementById("homeButton");

// Función para hacer scroll hacia arriba al hacer clic en Home
homeButton.addEventListener("click", (event) => {
    event.preventDefault();  // Evitar el comportamiento predeterminado del enlace
    window.scrollTo({
        top: 0,               // Desplazar hacia el inicio de la página
        behavior: "smooth"     // Desplazamiento suave
    });
});
