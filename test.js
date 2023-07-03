

async function readJson(url) {
    try {
        const response = await fetch(url);
        const jsonData = await response.json();
        return jsonData;
    } catch (error) {
        console.error(error);
        return null;
    }
}

const folder_patterns = 'website/images/patterns_json/'
const path_names=folder_patterns +'patterns.json'

a=readJson(path_names)



async function create_interaction(side_lenght) {

    const patterns = await readJson(path_names)

    let interaction = new Array(side_lenght);

    for (let i = 0; i < patterns.length; i++) {
        const pattern = await readJson(folder_patterns + patterns[i]);
        for (let col = 0; col < side_lenght; col++) {
            if (i === 0) interaction[col] = new Array(side_lenght);

            for (let row = 0; row < side_lenght; row++) {
                if (i === 0) interaction[col][row] = 0;
                interaction[col][row] += pattern[col][row];
            }
        }
    }
    return interaction
}


console.log(readJson(path_names))

