#include <GLES2/gl2.h>
#include <GLFW/glfw3.h>
#include <emscripten/emscripten.h>
#include <cstdio>

static const char *VERT_SRC = R"(
attribute vec2 pos;
varying vec2 v_uv;
void main() {
    v_uv = pos * 0.5 + 0.5;
    gl_Position = vec4(pos, 0.0, 1.0);
}
)";

static const char *FRAG_SRC = R"(
precision mediump float;
varying vec2 v_uv;
uniform float u_time;
void main() {
    vec2 p = v_uv - 0.5;
    float dist = length(p) * 2.0;
    float ripple = sin(10.0 * dist - u_time * 2.0);
    float alpha = smoothstep(0.5, 0.3, dist) + 0.5 * ripple;
    vec3 color = vec3(0.2, 0.7, 1.0);
    gl_FragColor = vec4(color, alpha);
}
)";

GLuint prog, vbo;
GLFWwindow* win;

GLuint compile(GLenum t,const char*src){
    GLuint s=glCreateShader(t);
    glShaderSource(s,1,&src,nullptr);
    glCompileShader(s);
    return s;
}

void init(){
    GLuint vs=compile(GL_VERTEX_SHADER,VERT_SRC);
    GLuint fs=compile(GL_FRAGMENT_SHADER,FRAG_SRC);
    prog=glCreateProgram();
    glAttachShader(prog,vs);
    glAttachShader(prog,fs);
    glLinkProgram(prog);

    float verts[]={-1,-1, 1,-1, -1,1, 1,1};
    glGenBuffers(1,&vbo);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    glBufferData(GL_ARRAY_BUFFER,sizeof(verts),verts,GL_STATIC_DRAW);
}

void render(){
    int w,h; glfwGetFramebufferSize(win,&w,&h);
    glViewport(0,0,w,h);
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(prog);
    glUniform1f(glGetUniformLocation(prog,"u_time"),(float)glfwGetTime());
    glBindBuffer(GL_ARRAY_BUFFER,vbo);
    GLint loc=glGetAttribLocation(prog,"pos");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc,2,GL_FLOAT,GL_FALSE,0,0);
    glDrawArrays(GL_TRIANGLE_STRIP,0,4);
    glfwSwapBuffers(win);
}

void loop(){
    glfwPollEvents();
    render();
}

int main(){
    if(!glfwInit()) return 1;
    win=glfwCreateWindow(800,600,"Glass",nullptr,nullptr);
    glfwMakeContextCurrent(win);
    init();
    emscripten_set_main_loop(loop,0,1);
    return 0;
}
