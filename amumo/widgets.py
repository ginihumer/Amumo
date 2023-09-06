# ipywidgets meets plotly: https://plotly.com/python/figurewidget-app/

import traitlets
from ipywidgets import widgets
import plotly.graph_objects as go
import io
from PIL import Image
import numpy as np

from sklearn.decomposition import PCA
from openTSNE import TSNE
from umap import UMAP
import torch

import math

from . import model
from .utils import get_textual_label_for_cluster, get_embedding, get_similarity, get_cluster_sorting, get_modality_distance, calculate_val_loss, get_closed_modality_gap, get_modality_gap_normed, l2_norm, get_gap_direction, get_closed_modality_gap_rotated, get_modality_distance_rotated


class SimilarityHeatmapWidget(widgets.VBox):
    
    value = traitlets.Any(np.zeros((6,6))).tag(sync=True)
    cluster = traitlets.Any().tag(sync=True)

    hover_idx = traitlets.List([]).tag(sync=True)


    def __init__(self, zmin=None, zmax=None):
        super(SimilarityHeatmapWidget, self).__init__()

        self.fig_widget = go.FigureWidget(data=[go.Heatmap(z=self.value, zmin=zmin, zmax=zmax)])
        self.heatmap = self.fig_widget.data[0]
        self.heatmap.hoverinfo = "text"
        self.fig_widget.update_layout(width=500, height=420,
            xaxis = dict(
                tickmode = 'array',
                tickvals = [len(self.value)/4, 3*len(self.value)/4],
                ticktext = ['Image', 'Text']
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = [len(self.value)/4, 3*len(self.value)/4],
                ticktext = ['Image', 'Text']
            ),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        self.fig_widget.update_yaxes(autorange='reversed', fixedrange=False)
        self.fig_widget.update_xaxes(fixedrange=False)
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

        self.children = [widgets.HBox([self.fig_widget])]



    def _get_matrix_gridlines(self):
        return [
            go.layout.Shape(type='line', x0=len(self.value)/2-0.5, y0=0-0.5, x1=len(self.value)/2-0.5, y1=len(self.value)-0.5, line=dict(color="black", width=1)),
            go.layout.Shape(type='line', y0=len(self.value)/2-0.5, x0=0-0.5, y1=len(self.value)/2-0.5, x1=len(self.value)-0.5, line=dict(color="black", width=1))
        ]


    @traitlets.validate("value")
    def _validate_value(self, proposal):
        # print("TODO: validate value")
        return proposal.value

    @traitlets.observe("value")
    def onUpdateValue(self, change):
        self.fig_widget.data[0].z = self.value
        self.fig_widget.layout.shapes = self._get_matrix_gridlines()

        self.fig_widget.update_layout(
            xaxis = dict(tickvals = [len(self.value)/4, 3*len(self.value)/4]),
            yaxis = dict(tickvals = [len(self.value)/4, 3*len(self.value)/4])
        )


    # @traitlets.validate("cluster")
    # def _validate_cluster(self, proposal):
        # takes a list of cluster labels + sizes
    #     print("TODO: validate cluster")
    #     return proposal.value

    @traitlets.observe("cluster")
    def onUpdateCluster(self, change):
        cluster_shapes = self._get_matrix_gridlines()
        labels, sizes = self.cluster
        offset = 0-0.5 # -0.5 because heatmap rectangles are drawn around [-0.5, 0.5]
        for (cluster_label, cluster_size) in zip(labels, sizes):
            if cluster_size > 5:
                textposition = 'middle left' if offset < len(self.value)/2/2 else 'middle right'

                # see https://plotly.com/python/shapes/
                cluster_shapes += [go.layout.Shape(
                    type='rect', 
                    x0=len(self.value)/2+offset, 
                    y0=offset, 
                    x1=len(self.value)/2+offset+cluster_size, 
                    y1=offset+cluster_size, 
                    label=dict(text=cluster_label, textposition=textposition, font=dict(size=10, color="white"), padding=np.log(cluster_size)*10), 
                    line=dict(width=1, color='white')
                )]

            offset += cluster_size
            
        self.fig_widget.layout.shapes = cluster_shapes


    @traitlets.observe("hover_idx")
    def onUpdateHoverIdx(self, change):
        shapes = [sh for sh in self.fig_widget.layout.shapes if sh.name != 'hover_idx' and sh.name != 'hover_idx']

        for (x_idx, y_idx) in self.hover_idx:
            if x_idx >= 0 and x_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', x0=x_idx, y0=0-0.5, x1=x_idx, y1=len(self.value)-0.5, line=dict(color="grey", width=1)))
            if y_idx >= 0 and y_idx < len(self.value):
                shapes.append(go.layout.Shape(name='hover_idx', type='line', y0=y_idx, x0=0-0.5, y1=y_idx, x1=len(self.value)-0.5, line=dict(color="grey", width=1)))
        
        self.fig_widget.layout.shapes = shapes



class HoverWidget(widgets.VBox):
    
    valueX = traitlets.Any().tag(sync=True)
    valueY = traitlets.Any().tag(sync=True)


    def __init__(self, width=300):
        super(HoverWidget, self).__init__()

        self.width = width

        output_dummy_img = io.BytesIO()
        Image.new('RGB', (self.width,self.width)).save(output_dummy_img, format="JPEG")
        self.img_widgets = {'valueX': widgets.Image(value=output_dummy_img.getvalue(), width=0, height=0), 
                            'valueY': widgets.Image(value=output_dummy_img.getvalue(), width=0)} #, height=0)}
        self.txt_widgets = {'valueX': widgets.HTML(value='', layout=widgets.Layout(width="%ipx"%self.width)), 
                            'valueY': widgets.HTML(value='', layout=widgets.Layout(width="%ipx"%self.width)), }
        
        self.children = [widgets.VBox(list(self.txt_widgets.values())), widgets.VBox(list(self.img_widgets.values()))]

        self.layout = widgets.Layout(width="%ipx"%(self.width+10), height="inherit")


    @traitlets.validate("value1", "value2")
    def _validate_value(self, proposal):
        # print("TODO: validate value1")
        return proposal.value

    @traitlets.observe("valueX", "valueY")
    def onUpdateValue(self, change):
        cur_img_widget = self.img_widgets[change.name]
        cur_txt_widget = self.txt_widgets[change.name]
        if type(change.new) is io.BytesIO:
            cur_img_widget.value = change.new.getvalue()
            cur_img_widget.width = self.width
            # cur_img_widget.height = self.width
            cur_txt_widget.value = ""
        else:
            cur_txt_widget.value = "<div style='word-wrap: break-word;'>{}</div>".format(change.new)
            cur_img_widget.width = 0
            # cur_img_widget.height = 0



available_projection_methods = {
    'PCA': {'module': PCA, 'OOS':False}, # OOS: flag to signal whether or not out of sample is possible
    'TSNE': {'module': TSNE, 'OOS':False},
    'UMAP': {'module': UMAP, 'OOS':True},
}

class ScatterPlotWidget(widgets.VBox):
    
    embedding = traitlets.Any().tag(sync=True)
    cluster = traitlets.Any().tag(sync=True)

    def __init__(self, seed=31415):
        super(ScatterPlotWidget, self).__init__()

        self.seed=seed

        self.nr_components_widget = widgets.BoundedIntText(
            value=2,
            min=2,
            max=10,
            step=1,
            description='Nr Components:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.nr_components_widget.observe(self.onUpdateValue, 'value')

        self.x_component_widget = widgets.BoundedIntText(
            value=1,
            min=1,
            max=10,
            step=1,
            description='X component:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.x_component_widget.observe(self.update_scatter, 'value')

        self.y_component_widget = widgets.BoundedIntText(
            value=2,
            min=1,
            max=10,
            step=1,
            description='Y component:',
            disabled=False,
            layout=widgets.Layout(width="150px")
        )
        self.y_component_widget.observe(self.update_scatter, 'value')

        traitlets.dlink((self.nr_components_widget, 'value'), (self.x_component_widget, 'max'))
        traitlets.dlink((self.nr_components_widget, 'value'), (self.y_component_widget, 'max'))


        self.fig_widget = go.FigureWidget()
        self.fig_widget.add_trace(go.Scatter(name = 'image', x=[0,1,2,3], y=[0,1,2,3], mode="markers", marker_color='blue'))
        self.fig_widget.add_trace(go.Scatter(name = 'text', x=[3,2,1,0], y=[0,1,2,3], mode="markers", marker_color='orange'))
        self.fig_widget.update_layout(width=400, 
                                      height=300, 
                                      margin=dict(l=10, r=10, t=10, b=10),
                                      legend=dict(
                                            yanchor="top",
                                            y=0.99,
                                            xanchor="left",
                                            x=0.01  
                                            )
                                      )

        self.scatter_image = self.fig_widget.data[0]
        self.scatter_text = self.fig_widget.data[1]

        self.select_projection_method = widgets.Dropdown(
            description='Method: ',
            value='PCA',
            options=list(available_projection_methods),
        )
        self.select_projection_method.observe(self._update_projection_method, 'value')

        self.use_oos_projection = widgets.Checkbox(
            value=False,
            description='Use out of sample projection',
            disabled=not available_projection_methods[self.select_projection_method.value]['OOS'],
            indent=False
        )
        self.use_oos_projection.observe(self._update_projection_method, 'value')

        
        self.children = [self.select_projection_method, self.use_oos_projection, self.nr_components_widget, widgets.HBox([self.x_component_widget, self.y_component_widget]), self.fig_widget]
        

    def _update_projection_method(self, change):
        print('', available_projection_methods[self.select_projection_method.value]['OOS'])
        self.use_oos_projection.disabled = not available_projection_methods[self.select_projection_method.value]['OOS']
        self.onUpdateValue(change)


    @traitlets.validate("embedding")
    def _validate_value(self, proposal):
        # print("TODO: validate embedding")
        return proposal.value

    @traitlets.observe("embedding")
    def onUpdateValue(self, change):
        projection_method = available_projection_methods[self.select_projection_method.value]
        # TODO: add UI for distance metric
        if self.select_projection_method.value == 'PCA':
            projection = projection_method['module'](n_components=self.nr_components_widget.value, random_state=self.seed)
        else:
            projection = projection_method['module'](n_components=self.nr_components_widget.value, metric="cosine", random_state=self.seed)
            

        if not self.use_oos_projection.disabled and self.use_oos_projection.value:
            project_by = 'image' # TODO: add user select for this
    
            if project_by == "image":
                self.image_embedding_projection = projection.fit_transform(self.embedding[:int(len(self.embedding)/2),:])
                self.text_embedding_projection = projection.transform(self.embedding[int(len(self.embedding)/2):,:])
            elif project_by == "text":
                self.text_embedding_projection = projection.fit_transform(self.embedding[int(len(self.embedding)/2):,:])
                self.image_embedding_projection = projection.transform(self.embedding[:int(len(self.embedding)/2),:])

            self.embedding_projection = np.concatenate((self.image_embedding_projection, self.text_embedding_projection))
            
        else:
            if self.select_projection_method.value == 'TSNE':
                self.embedding_projection = projection.fit(self.embedding)
            else:
                self.embedding_projection = projection.fit_transform(self.embedding)

            self.image_embedding_projection = self.embedding_projection[:int(len(self.embedding)/2),:]
            self.text_embedding_projection = self.embedding_projection[int(len(self.embedding)/2):,:]

        self.update_scatter(change)

    def update_scatter(self, change):
        self.scatter_image.x = self.image_embedding_projection[:,self.x_component_widget.value-1]
        self.scatter_image.y = self.image_embedding_projection[:,self.y_component_widget.value-1]

        self.scatter_text.x = self.text_embedding_projection[:,self.x_component_widget.value-1]
        self.scatter_text.y = self.text_embedding_projection[:,self.y_component_widget.value-1]

        lines = []

        for line_idx in range(len(self.text_embedding_projection)):
            lines.append(go.layout.Shape(name='hover_idx', 
                                         type='line', 
                                         x0=self.image_embedding_projection[line_idx,self.x_component_widget.value-1], 
                                         y0=self.image_embedding_projection[line_idx,self.y_component_widget.value-1], 
                                         x1=self.text_embedding_projection[line_idx,self.x_component_widget.value-1], 
                                         y1=self.text_embedding_projection[line_idx,self.y_component_widget.value-1], 
                                         line=dict(color="grey", width=1)))
            # plt.plot((image_embedding_pca[line_idx,0], text_embedding_pca[line_idx,0]), (image_embedding_pca[line_idx,1], text_embedding_pca[line_idx,1]), 'black', linestyle='-', marker='', linewidth=1, alpha=0.2)

        self.fig_widget.layout.shapes = lines


    @traitlets.validate("cluster")
    def _validate_cluster(self, proposal):
        # takes a list of cluster labels + sizes
        # print("TODO: validate cluster")
        return proposal.value

    @traitlets.observe("cluster")
    def onUpdateCluster(self, change):
        print(change)



class CLIPExplorerWidget(widgets.AppLayout):
    idcs = traitlets.Any().tag(sync=True)

    def __init__(self, dataset_name, all_images, all_prompts, models=None):
        ### models... list of strings or instances that inherit from CLIPModelInterface 
        super(CLIPExplorerWidget, self).__init__()

        if models is None:
            models = model.available_CLIP_models

        self.models = {}
        for m in models:
            if type(m) == str:
                self.models[m] = model.get_model(m)
            elif issubclass(type(m), model.CLIPModelInterface):
                self.models[m.model_name] = m
            else:
                print('skipped', m, 'because it is not string or of type CLIPModelInterface')

        
        self.dataset_name = dataset_name
        self.all_images = np.array(all_images)
        self.all_prompts = np.array(all_prompts)
        self.size = len(all_images)
        self.idcs = np.arange(self.size)

        # ui select widgets
        self.model_select_widget = widgets.Dropdown(
            description='Model: ',
            value=list(self.models.keys())[0],
            options=list(self.models.keys()),
        )

        self.cluster_similarity_matrix_widget = widgets.Checkbox(
            value=False,
            description='Cluster matrix by similarity',
            disabled=False,
            indent=False
        )

        self.close_modality_gap_widget = widgets.Checkbox(
            value=False,
            description='Close modality gap',
            disabled=False,
            indent=False
        )

        # output widgets
        self.hover_widget = HoverWidget()

        m = self.models[self.model_select_widget.value]
        image_embedding_norm, text_embedding_norm, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)
        self.scatter_widget = ScatterPlotWidget()
        self.scatter_widget.embedding = np.concatenate((image_embedding_norm, text_embedding_norm))

        modality_distance = get_modality_distance(image_embedding_norm, text_embedding_norm)
        validation_loss = calculate_val_loss(image_embedding_norm, text_embedding_norm, logit_scale.exp())
        self.log_widget = widgets.Output()
        with self.log_widget:
            print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))

        similarity_texts_images, similarity_all = get_similarity(image_embedding_norm, text_embedding_norm)
        self.heatmap_widget = SimilarityHeatmapWidget()
        self.heatmap_widget.value = similarity_all



        # callback functions
        self.model_select_widget.observe(self.model_changed, names="value")
        self.cluster_similarity_matrix_widget.observe(self.model_changed, names='value')
        self.close_modality_gap_widget.observe(self.model_changed, names='value')
        self.heatmap_widget.heatmap.on_hover(self.hover_fn)
        self.scatter_widget.scatter_image.on_hover(self.scatter_hover_fn)
        self.scatter_widget.scatter_text.on_hover(self.scatter_hover_fn)
        self.scatter_widget.scatter_image.on_unhover(self.scatter_unhover_fn)
        self.scatter_widget.scatter_text.on_unhover(self.scatter_unhover_fn)

        # display everyting
        self.header = widgets.VBox([widgets.HBox([self.model_select_widget, self.cluster_similarity_matrix_widget, self.close_modality_gap_widget]),self.log_widget])
        self.header.layout.height = '80px'
        vis_widgets = widgets.HBox([self.heatmap_widget, self.scatter_widget])
        self.center = vis_widgets
        self.right_sidebar = self.hover_widget
        self.height = '700px'


    def model_changed(self, change):

        self.log_widget.clear_output()
        with self.log_widget:
            print("loading...")

        m = self.models[self.model_select_widget.value]
        image_embedding, text_embedding, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)

        if self.close_modality_gap_widget.value:
            image_embedding, text_embedding = get_closed_modality_gap(image_embedding, text_embedding)
            # image_embedding, text_embedding = get_closed_modality_gap_rotated(image_embedding, text_embedding)

        self.scatter_widget.embedding = np.concatenate((image_embedding, text_embedding))

        modality_distance = get_modality_distance(image_embedding, text_embedding)
        # modality_distance = get_modality_distance_rotated(image_embedding, text_embedding)
        
        validation_loss = calculate_val_loss(image_embedding, text_embedding, logit_scale.exp())
        
        similarity_texts_images, similarity_all = get_similarity(image_embedding, text_embedding)

        cluster_labels = []
        cluster_sizes = []

        if self.cluster_similarity_matrix_widget.value:
            self.idcs, clusters, clusters_unsorted = get_cluster_sorting(similarity_texts_images)
            for c in set(clusters):
                cluster_size = np.count_nonzero(clusters==c)
                cluster_label = get_textual_label_for_cluster(np.where(clusters_unsorted==c)[0], self.all_prompts)
                cluster_labels.append(cluster_label)
                cluster_sizes.append(cluster_size)
        else:
            self.idcs = np.arange(self.size)

        # with heatmap_widget.batch_update():
        matrix_sort_idcs = np.concatenate([self.idcs, self.idcs+self.size], axis=0) # need to do double index because we combined images and texts
        self.heatmap_widget.value = similarity_all[matrix_sort_idcs, :][:, matrix_sort_idcs]
        self.heatmap_widget.cluster = (cluster_labels, cluster_sizes)

        self.log_widget.clear_output()
        with self.log_widget:
            print('Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))


    def hover_fn(self, trace, points, state):
        x_idx = points.xs[0]
        y_idx = points.ys[0]
            
        self.heatmap_widget.hover_idx = [(x_idx, y_idx)]

        if x_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[self.idcs][x_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueX = output_img
        else:
            self.hover_widget.valueX = self.all_prompts[self.idcs][x_idx%self.size]
        
        if y_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[self.idcs][y_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueY = output_img
        else:
            self.hover_widget.valueY = self.all_prompts[self.idcs][y_idx%self.size]

    def scatter_hover_fn(self, trace, points, state):
        if len(points.point_inds) < 1:
            return
        idx = points.point_inds[0]
        # print(trace.name, idx) # image vs text trace

        self.hover_widget.valueX = self.all_prompts[idx]

        output_img = io.BytesIO()
        self.all_images[idx].resize((300,300)).save(output_img, format='JPEG')
        self.hover_widget.valueY = output_img
        
        inverse_idcs = np.argsort(self.idcs)
        heatmap_idx = inverse_idcs[idx]
        self.heatmap_widget.hover_idx = [(heatmap_idx, self.size + heatmap_idx), (self.size + heatmap_idx, heatmap_idx)]


    def scatter_unhover_fn(self, trace, points, state):
        self.heatmap_widget.hover_idx = []




class CLIPComparerWidget(widgets.AppLayout):


    def __init__(self, dataset_name, all_images, all_prompts, models=list(model.available_CLIP_models), close_modality_gap=False):
        super(CLIPComparerWidget, self).__init__()
        # close_modality_gap: boolean or list of booleans with same length as models that specifies whether or not the modality gap should be closed
        ### models... list of strings or instances that inherit from CLIPModelInterface 

        if models is None:
            models = model.available_CLIP_models

        self.models = {}
        for m in models:
            if type(m) == str:
                self.models[m] = model.get_model(m)
            elif issubclass(type(m), model.CLIPModelInterface):
                self.models[m.model_name] = m
            else:
                print('skipped', m, 'because it is not string or of type CLIPModelInterface')

        

        if type(close_modality_gap) == bool:
            close_modality_gap = [close_modality_gap] * len(self.models)

        assert type(close_modality_gap) == list and len(close_modality_gap) == len(self.models), 'close_modality_gap must be a bool or list of the same length as models'
        
        self.dataset_name = dataset_name
        self.all_images = np.array(all_images)
        self.all_prompts = np.array(all_prompts)
        self.size = len(all_images)
        
        # output widgets
        self.hover_widget = HoverWidget()

        self.heatmap_grid = widgets.GridspecLayout(math.ceil(len(self.models)/2), 2)
        for i in range(len(self.models.keys())):
            key = list(self.models.keys())[i]
            m = self.models[key]

            image_embedding_norm, text_embedding_norm, logit_scale = get_embedding(m, self.dataset_name, self.all_images, self.all_prompts)
            if close_modality_gap[i]:
                image_embedding_norm, text_embedding_norm = get_closed_modality_gap(image_embedding_norm, text_embedding_norm)

            _, similarity_all = get_similarity(image_embedding_norm, text_embedding_norm)
            heatmap_widget = SimilarityHeatmapWidget()#(zmin=0, zmax=1)
            heatmap_widget.value = similarity_all
            heatmap_widget.heatmap.on_hover(self.hover_fn)
            # heatmap_widget.layout = widgets.Layout(height='500px', width='auto')

            modality_distance = get_modality_distance(image_embedding_norm, text_embedding_norm)
            validation_loss = calculate_val_loss(image_embedding_norm, text_embedding_norm, logit_scale.exp())
            text_widget = widgets.HTML(value='<h2>' + m.model_name + '</h2>' 
                                       + ' Modality distance: %.2f | Loss: %.2f'%(modality_distance, validation_loss))

            self.heatmap_grid[int(i/2), i%2] = widgets.VBox([text_widget, heatmap_widget])
            self.heatmap_grid[int(i/2), i%2].layout.height = '500px'

        self.right_sidebar = self.hover_widget
        self.center = self.heatmap_grid
        self.height = '900px'


    def hover_fn(self, trace, points, state):
        x_idx = points.xs[0]
        y_idx = points.ys[0]
            
        for c in self.heatmap_grid.children:
            c.children[1].hover_idx = [(x_idx, y_idx)]

        if x_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[x_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueX = output_img
        else:
            self.hover_widget.valueX = self.all_prompts[x_idx%self.size]
        
        if y_idx < self.size:
            output_img = io.BytesIO()
            self.all_images[y_idx].resize((300,300)).save(output_img, format='JPEG')
            self.hover_widget.valueY = output_img
        else:
            self.hover_widget.valueY = self.all_prompts[y_idx%self.size]


class ModalityGapWidget(widgets.AppLayout):
    
    def __init__(self, image_embedding, text_embedding, logit_scale=100.0, title='Loss Landscape CLIP'):
        super(ModalityGapWidget, self).__init__()
        
        image_embedding = np.array(image_embedding)
        text_embedding = np.array(text_embedding)

        modality_gap = get_modality_gap_normed(image_embedding, text_embedding)
        
        distance_lst = []
        loss_lst = []
        for delta in np.arange(-5.0, 5.0, 0.25): 
            modified_text_features = l2_norm(text_embedding) + 0.5 * delta * modality_gap
            modified_text_features = l2_norm(modified_text_features)

            modified_image_features = l2_norm(image_embedding) - 0.5 * delta * modality_gap
            modified_image_features = l2_norm(modified_image_features)

            avg_val_loss = calculate_val_loss(torch.from_numpy(modified_image_features), torch.from_numpy(modified_text_features), logit_scale = logit_scale)

            pca = PCA(n_components=6)
            pca.fit(np.concatenate((image_embedding, text_embedding), axis=0))

            gap_direction = get_gap_direction(modified_image_features, modified_text_features, pca)

            loss_lst.append(avg_val_loss)

            # Euclidean distance between mass centers
            distance_lst.append(
                get_modality_distance(modified_image_features, modified_text_features) * gap_direction
            )


        orig_distance = get_modality_distance(image_embedding, text_embedding)

        fig = go.FigureWidget(data=go.Scatter(x=distance_lst, y=loss_lst, mode='lines+markers', hovertemplate='Distance: %{x} <br>Loss: %{y}'))
        fig.add_shape(type="line",
            x0=orig_distance, y0=0, x1=orig_distance, y1=max(loss_lst)*1.2,
            line=dict(
                color="Black",
                width=1,
                dash="dash",
            )
        )
        fig.update_layout(xaxis_title='Euclidean Distance', yaxis_title='Loss', width=500, title=title)
        
        self.center = widgets.HBox([fig])
        