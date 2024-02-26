
var interpolate = d3.scaleSequential(d3.interpolatePlasma);
var plasma_colors = [];
dark = d3.lab(interpolate(0)).l < 50;
for (let i = 0; i < 100; ++i) {
    plasma_colors.push([i / 99, d3.rgb(interpolate(i / (99))).hex()]);
}

// load data
const mscoco_val_prompts_promise = fetch('./exported_data/MSCOCO-Val_size-100/prompts.txt')
    .then(response => response.text())
    .then(data => data.split('\n'))
    .catch(error => console.error(error));

const mscoco_val_projections_promise = d3.csv("./exported_data/MSCOCO-Val_size-100/projections.csv");


const clip_loss_landscape_promise = fetch('./exported_data/MSCOCO-Val_size-5000/CLIP_loss_landscape.json')
    .then(response => response.json())
    .catch(error => console.error(error));

const cyclip_loss_landscape_promise = fetch('./exported_data/MSCOCO-Val_size-5000/CyCLIP_loss_landscape.json')
    .then(response => response.json())
    .catch(error => console.error(error));

const load_similarities_fn = function (dataset, name) {
    return fetch('./exported_data/' + dataset + '/similarities/' + name + '.csv')
        .then(response => response.text())
        .then(csvData => {
            const rows = csvData.split('\n');
            const data = rows.filter(row => row.length > 1).map(row => row.split(','));
            return data;
        })
        .catch(error => console.error(error));
}

const load_meta_info_fn = function (dataset, name) {
    return fetch('./exported_data/' + dataset + '/similarities/' + name + '_meta_info.json')
        .then(response => response.json())
        .catch(error => console.error(error));
}

