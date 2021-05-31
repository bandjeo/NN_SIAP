idGenerator = () => {
    id = 0
    generator = () => {
        id+=1;
        return id
    }
    return generator
}

get_next = idGenerator();
var ingredients = []

removeListItem = (ingredient_id) => {
    var elem = document.getElementById(`li_${ingredient_id}`);
    elem && elem.parentNode.removeChild(elem);
}

addIngredient = (name) => {
    ingredient_id = get_next();
    ingredients.push({
        ingredient_id,
        name
    })
    document.getElementById('itemgroup').innerHTML += `<li id="li_${ingredient_id}" class="list-group-item d-flex justify-content-between align-items-center">${name}<button id="ingredient_${ingredient_id}" class="ingredient_delete btn btn-danger"><span id="ingredientspan_${ingredient_id}" class="fa fa-trash"></span></button></li>`
    
    for (let i of document.querySelectorAll(`.ingredient_delete`)){
        i.addEventListener('click', (event) => {
            ingredient_id = +event.target.id.split('_')[1]
            removeListItem(ingredient_id);
            ingredients = ingredients.filter(e => e.ingredient_id != ingredient_id);
        })
    }
}

document.getElementById('addbtn').addEventListener('click', (event) => {
    event.preventDefault();
    value = document.getElementById('ingredient').value;
    if (value)
    {
        addIngredient(value);
        document.getElementById('ingredient').value = '';
    } 
})

