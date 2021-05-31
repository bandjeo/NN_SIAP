
var current_next = 3;
var x_interval = null;


function changeLobster() {
    if (current_next == 3){
        document.getElementById('lobster').setAttribute('src', '/public/3.png')
        current_next = 2;
    }else{
        document.getElementById('lobster').setAttribute('src', '/public/2.png')
        current_next = 3;
    }
}

function spawnWaitTab() {
    document.getElementById('searchTab').style.opacity = 0;
    setTimeout(() => {
        document.getElementById('searchTab').style.display = 'none';
        document.getElementById('waitTab').style.display = 'block';
        setTimeout(() => {
            document.getElementById('waitTab').style.opacity = 1;
            setTimeout(() => {
                x_interval = setInterval(changeLobster, 450)
            }, 250)
        }, 10)
    }, 500)
}

async function getAndShowInstructions() {
    instruction = (await axios.post('/recipe', {'ingredients': ingredients.map(e => e.name)})).data;

    document.getElementById('instruction_container').innerHTML = instruction.split('.').filter(e => e!='').map(e => `<span class="sentence border border-light border-2">${e}.</span>`).join(' ');
    document.getElementById('ingredients_container').innerHTML = ingredients.map(e => `<p> ${e.name}</p>`).join(' ');

    clearInterval(x_interval)
    
    document.getElementById('lobster').style.left = 'calc(50% - 35px - 10px)';
    document.getElementById('lobster').style.top = '40';
    document.getElementById('lobster').style.width = '70px';
    document.getElementById('plate').style.left = 'calc(50% - 75px)';
    document.getElementById('plate').style.top = '0';    
    document.getElementById('plate').style.width = '150px';
    // document.getElementById('waitTab').style.opacity = 0;
    setTimeout(() => {
        document.getElementById('instructionsTab').style.display = 'block';
        // document.getElementById('waitTab').style.display = 'none';
        setTimeout(() => {
            document.getElementById('instructionsTab').style.opacity = 1;
            document.getElementById('lobster').style.cursor = 'pointer';
            document.getElementById('lobster').addEventListener('click', () => document.location.reload())
            document.getElementById('plate').style.cursor = 'pointer';
            document.getElementById('plate').addEventListener('click', () => document.location.reload())

        }, 10);
    }, 500);
}

document.getElementById('generatebtn').addEventListener('click', (event) => {
    spawnWaitTab();
    setTimeout(async () => {
        getAndShowInstructions();
    }, 2000);
    

})

