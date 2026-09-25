#include "Camera.hpp"

OrthographicCamera::OrthographicCamera(float_t left, float_t right,
                                       float_t bottom, float_t top,
                                       float_t near, float_t far,
                                       glm::vec3 position)
    : projection_matrix(glm::ortho(left, right, bottom, top, near, far)),
      view_matrix(glm::mat4(1.0f)) {
  near_plane = near;
  far_plane = far;
  frustum = glm::vec4(left, right, bottom, top);
  setPosition(position);
  projection_view_matrix = projection_matrix * view_matrix;
}

void OrthographicCamera::init(float_t left, float_t right, float_t bottom,
                              float_t top, float_t near, float_t far,
                              glm::vec3 position) {
  near_plane = near;
  far_plane = far;
  frustum_size = top;
  default_frustum_size = top;
  zoom = 0.0f;
  frustum = glm::vec4(left, right, bottom, top);
  projection_matrix = glm::ortho(left, right, bottom, top, near, far);
  this->setPosition(position);
  projection_view_matrix = projection_matrix * view_matrix;
}

void OrthographicCamera::setPosition(glm::vec3& position) {
  this->position = position;
  this->recalculateViewMatrix();
}

void OrthographicCamera::setFrustum(glm::vec4& frustum) {
  this->frustum = frustum;
  recalculateProjectionMatrix();
}

void OrthographicCamera::zoomIn(FrameHandler& input) {
  zoom +=
      properties.far_speed + (properties.close_speed - properties.far_speed) *
                                 ((zoom - properties.min_zoom) /
                                  (properties.max_zoom - properties.min_zoom));
  if (zoom > properties.max_zoom) zoom = properties.max_zoom;
  position.x += input.mouse.cursor.current.x * frustum_size;
  position.y += input.mouse.cursor.current.y * frustum_size;
  frustum_size = default_frustum_size - zoom;
  frustum.x = 0.0f;
  frustum.y = frustum_size;
  frustum.z = 0.0f;
  frustum.w = frustum_size;

  position.x -= frustum_size * input.mouse.cursor.current.x;
  position.y -= frustum_size * input.mouse.cursor.current.y;
  /*position.x = fmax(0.0f, position.x);
  position.y = fmax(0.0f, position.y);
  if (position.x + frustum_size > 1.0f) position.x -= (position.x +
  frustum_size) - 1.0f; if (position.y + frustum_size > 1.0f) position.y -=
  (position.y + frustum_size) - 1.0f;*/
  frustum_ratio = input.properties.scr_width / frustum_size;

  this->recalculateProjectionViewMatrix();
}

void OrthographicCamera::zoomOut(FrameHandler& input) {
  zoom -=
      properties.far_speed + (properties.close_speed - properties.far_speed) *
                                 ((zoom - properties.min_zoom) /
                                  (properties.max_zoom - properties.min_zoom));
  if (zoom < properties.min_zoom) zoom = properties.min_zoom;

  position.x += input.mouse.cursor.current.x * frustum_size;
  position.y += input.mouse.cursor.current.y * frustum_size;
  frustum_size = default_frustum_size - zoom;
  frustum.x = 0.0f;
  frustum.y = frustum_size;
  frustum.z = 0.0f;
  frustum.w = frustum_size;

  position.x -= frustum_size * input.mouse.cursor.current.x;
  position.y -= frustum_size * input.mouse.cursor.current.y;
  position.x = fmax(0.0f, position.x);
  position.y = fmax(0.0f, position.y);
  if (position.x + frustum_size > 1.0f)
    position.x -= (position.x + frustum_size) - 1.0f;
  if (position.y + frustum_size > 1.0f)
    position.y -= (position.y + frustum_size) - 1.0f;
  frustum_ratio = input.properties.scr_width / frustum_size;
  this->recalculateProjectionViewMatrix();
}

// void OrthographicCamera::updateZoom()
//{
//	if (InputHandler::mouse.scroll_direction == scroll_up) {
//		zoom += properties.far_speed + (properties.close_speed -
//properties.far_speed) * ((zoom - properties.min_zoom) / (properties.max_zoom -
//properties.min_zoom));
//	}
//	else {
//		zoom -= properties.far_speed + (properties.close_speed -
//properties.far_speed) * ((zoom - properties.min_zoom) / (properties.max_zoom -
//properties.min_zoom));
//	}
//	if (zoom < properties.min_zoom) zoom = properties.min_zoom;
//	if (zoom > properties.max_zoom) zoom = properties.max_zoom;
//
//	position.x += InputHandler::mouse.current_position.x * frustum_size;
//	position.y += InputHandler::mouse.current_position.y * frustum_size;
//	frustum_size = default_frustum_size - zoom;
//	frustum.x = 0.0f;
//	frustum.y = frustum_size;
//	frustum.z = 0.0f;
//	frustum.w = frustum_size;
//
//	position.x -= frustum_size * InputHandler::mouse.current_position.x;
//	position.y -= frustum_size * InputHandler::mouse.current_position.y;
//	position.x = fmax(0.0f, position.x);
//	position.y = fmax(0.0f, position.y);
//	if (position.x + frustum_size > 1.0f) position.x -= (position.x +
//frustum_size) - 1.0f; 	if (position.y + frustum_size > 1.0f) position.y -=
//(position.y + frustum_size) - 1.0f;
//
//	this->recalculateProjectionViewMatrix();
// }

void OrthographicCamera::updatePosition(FrameHandler& input) {
  position.x -= input.mouse.cursor.delta.x * frustum_size;
  position.y -= input.mouse.cursor.delta.y * frustum_size;
  if (position.x < 0.0f) position.x = 0.0f;
  if (position.x + frustum_size > 1.0f)
    position.x -= (position.x + frustum_size) - 1.0f;
  if (position.y < 0.0f) position.y = 0.0f;
  if (position.y + frustum_size > 1.0f)
    position.y -= (position.y + frustum_size) - 1.0f;
  this->recalculateViewMatrix();
}

void OrthographicCamera::recalculateProjectionMatrix() {
  projection_matrix = glm::ortho(frustum.x, frustum.y, frustum.z, frustum.w,
                                 near_plane, far_plane);
  projection_view_matrix = projection_matrix * view_matrix;
}

void OrthographicCamera::recalculateProjectionViewMatrix() {
  projection_matrix = glm::ortho(frustum.x, frustum.y, frustum.z, frustum.w,
                                 near_plane, far_plane);
  view_matrix = glm::inverse(glm::translate(glm::mat4(1.0f), position));
  projection_view_matrix = projection_matrix * view_matrix;
}

void OrthographicCamera::recalculateViewMatrix() {
  view_matrix = glm::inverse(glm::translate(glm::mat4(1.0f), position));
  projection_view_matrix = projection_matrix * view_matrix;
}

void OrthographicCamera::load(Properties& app_properties) {
  float_t left = 0.0f;
  float_t right = 1.0f;
  float_t bottom = 0.0f;
  float_t top = 1.0f;
  float_t near = -1.0f;
  float_t far = 1.0f;
  glm::vec3 start_position(0.0f, 0.0f, 0.0f);

  this->properties.min_zoom = 0.0f;
  this->properties.max_zoom = 0.99f;
  this->properties.far_speed = 0.1f;
  this->properties.close_speed = 0.00001f;

  this->frustum_ratio = app_properties.scr_width / top;
  init(left, right, bottom, top, near, far, start_position);
}