var image_id = 0;

function loadAsArrayBuffer(file_path, theFile) {
    var reader = new FileReader();

    reader.onload = function(loadedEvent) {
        dataView = loadedEvent.target.result;

        let select_block = document.getElementById('select-block'),
            preview_block = document.getElementById('preview-block'),
            image = document.getElementById('preview-image');
        image.src = dataView;

        select_block.style.display = 'none';
        preview_block.style.display = 'block';

        image.onload = function (){
            console.log(image.naturalWidth);
            if (image.naturalWidth > 446) {
                image.style.width = 446 + 'px';
            }
        }

        fetch_image(dataView)
    };

    reader.readAsDataURL(theFile);
}

function randomInt(min, max) {
	return min + Math.floor((max - min) * Math.random());
}

function upload_file(event){
    let file_path = event.target.value,
        image = event.target.files[0];
    loadAsArrayBuffer(file_path, image);
}

function fetch_image(data){
    image_id = randomInt(0, 1000000);
    let promise = fetch('/process', {
        body: JSON.stringify({
            'id': image_id,
            'image': data
        }),
        headers: {
            'Content-Type': 'application/json',
        },
        method: 'POST'
    }).then(response => response.json());
    console.log('Get promise:' + promise);
    promise.then(
            result => {
                if (result['success']) {
                    let data = result['payload'],
                        message = document.getElementById('message');
                        answer_block = document.getElementById('right-answer');
                        select = document.getElementById('selected-answer');
                    answer_block.style.display = 'block';
                    select.textContent = '';
                    message.style.whiteSpace = 'pre-line';
                    message.textContent = "Нейросеть считает,\n что портрет нарисован:";
                    for (let i = 0; i < data['names'].length; i++) {
                        var option = document.createElement("option");
                        option.text = data['names'][i];
                        option.value = data['names'][i];
                        select.appendChild(option);
                        message.textContent += "\n" + data['names'][i] + " (" + data['predictions'][i] + "%)";
                    }
                    message.textContent += "\n\nЗнаете, чем он нарисован?\nРасскажите нам:";
                }
                },
            error => {
                throw error;
            }
        );
}

function fetchAnswer() {
    let option = document.getElementById('selected-answer').value,
        answer_block = document.getElementById('right-answer'),
        message = document.getElementById('message');

    let promise = fetch('/answer', {
            body: JSON.stringify({
                "id": image_id,
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