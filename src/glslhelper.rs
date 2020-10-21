pub struct GLSLCompile {
    code: String,
    compiler: shaderc::Compiler,
}

impl GLSLCompile {
    pub fn new(code: &str) -> Self {
        GLSLCompile {
            code: code.to_string(),
            compiler: shaderc::Compiler::new().unwrap()
        }
    }
    pub fn compile(&mut self) -> Result<Vec<u32>, ()>{
        let bin = self.compiler.compile_into_spirv(
            &self.code,
            shaderc::ShaderKind::Compute,
            "name",
            "name",
            None,
        ).unwrap();
        Ok(bin.as_binary().to_vec())
    }
}

pub struct GLSLMatrix {
    code: String
}

impl GLSLBuilder for GLSLMatrix {
    fn new() -> Self {
        GLSLMatrix{code: "
            layout(set = 0, binding=0) buffer TMatrix {
                uint n;
                uint m;
                uint data[];
            };
        ".to_string()}
    }
    fn gen(&self) -> String {
        self.version() + &self.code
    }
}

pub trait GLSLBuilder {
    fn new() -> Self;
    fn gen(&self) -> String;
    fn version(&self) -> String {
        "#version 450".to_string()
    }
}