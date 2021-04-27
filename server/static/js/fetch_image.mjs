export function loadAsArrayBuffer(theFile) {
    var reader = new FileReader();

    reader.onload = function(loadedEvent) {
        var dataView = loadedEvent.target.result;

        let select_button = document.getElementById('select-button'),
            preview_block = document.getElementById('preview-block'),
            image = document.getElementById('preview-image'),
            loading = document.getElementById('loading'),
            message = document.getElementById('message');
        image.src = dataView;
        image.style.filter = "blur(3px)";
        message.style.whiteSpace = 'pre-line';
        message.textContent = 'Подождите немного\nИдёт обработка...'

        select_button.style.display = 'none';
        preview_block.style.display = 'block';

        image.onload = function (){
            if (image.naturalWidth > 446) {
                image.style.width = 446 + 'px';
                loading.style.left = 148 + 'px';
                loading.style.top = (image.naturalHeight * 223) / image.naturalWidth  - 75 + 'px';
            } else {
                if (image.naturalWidth < 150) {
                    image.style.width = 150 + 'px';
                    loading.style.left = 0 + 'px';
                    loading.style.top = (image.naturalHeight * 50) / image.naturalWidth  - 75 + 'px';
                } else {
                    loading.style.left = image.naturalWidth / 2 - 75 + 'px';
                    loading.style.top = image.naturalHeight / 2 - 75 + 'px';
                }
            }
        loading.style.display = 'block';
        }


        fetch_image(dataView);
    };

    reader.readAsDataURL(theFile);

    return true;
}

function randomInt(min, max) {
	return min + Math.floor((max - min) * Math.random());
}

function fetch_image(data){
    let image_id = randomInt(0, 1000000);

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
                        message = document.getElementById('message'),
                        loading = document.getElementById('loading'),
                        answer_block = document.getElementById('right-answer'),
                        image = document.getElementById('preview-image'),
                        select = document.getElementById('selected-answer');
                    image.style.filter = '';
                    answer_block.style.display = 'block';
                    select.textContent = '';
                    message.style.whiteSpace = 'pre-line';
                    loading.style.display = 'none';
                    message.textContent = "Нейросеть считает,\n что портрет нарисован:";
                    for (let i = 0; i < data['names'].length; i++) {
                        var option = document.createElement("option");
                        option.text = data['names'][i];
                        option.value = data['names'][i];
                        select.appendChild(option);
                        message.textContent += "\n" + data['names'][i] + " (" + data['predictions'][i] + "%)";
                    }
                    message.textContent += "\n\nЗнаете, чем он нарисован?\nРасскажите нам:";

                    image.setAttribute("image_id", image_id);
                }
                },
            error => {
                throw error;
            }
        );
}