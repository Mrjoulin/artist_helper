import { loadAsArrayBuffer } from './fetch_image.mjs';

let upload = document.getElementById('upload')

upload.addEventListener("change", upload_file, false)

function upload_file(event){
    let image = event.target.files[0];
    loadAsArrayBuffer(image);
    console.log("New image Id: " + document.getElementById('preview-image').getAttribute("image_id"));
}


function fetchAnswer() {
    let option = document.getElementById('selected-answer').value,
        answer_block = document.getElementById('right-answer'),
        message = document.getElementById('message'),
        image = document.getElementById('preview-image');

    image_id = image.getAttribute("image_id");

    if (!image_id) {
        console.log("Image ID not found");
        return;
    }

    let promise = fetch('/answer', {
            body: JSON.stringify({
                "id": +image_id,
                "answer": option
            }),
            headers: {
                'Content-Type': 'application/json',
            },
            method: 'POST'
        }).then(response => response.json());

    promise.then(
        result => {
            if (!result['success']) {
                alert('Got error on server')
            }
        },
        error => {
            throw error;
        }
    );

    answer_block.style.display = 'none';
    message.textContent = message.textContent.split('\n\n')[0] + '\n\nСпасибо за ответ';
}