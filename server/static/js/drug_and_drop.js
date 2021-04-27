import { loadAsArrayBuffer } from './fetch_image.mjs';

let dropArea = document.getElementById('select-block')

;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, preventDefaults, false)
})

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

;['dragenter', 'dragover'].forEach(eventName => {
  dropArea.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
  dropArea.addEventListener(eventName, unhighlight, false)
})

function highlight(e) {
  dropArea.classList.add('highlight')
}

function unhighlight(e) {
  dropArea.classList.remove('highlight')
}

dropArea.addEventListener('drop', handleDrop, false)

function handleDrop(e) {
    e.preventDefault();

    let file = e.dataTransfer.files[0];

    if (file) {
        loadAsArrayBuffer(file);

        console.log("New image Id: " + document.getElementById('preview-image').getAttribute("image_id"));
    } else {
        console.log("File not found");

        let item = e.dataTransfer.items[0];

        // TODO: Load image from url
    }
}